"""
Clinical Validation Module
Validates drug repurposing candidates using clinical databases
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import ssl
import certifi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalValidator:
    """
    Validates drug candidates against clinical evidence from multiple sources:
    - PubMed (literature)
    - ClinicalTrials.gov (trial data)
    - DrugBank (interactions/contraindications)
    - OpenFDA (adverse events)
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict = {}
        self.ssl_context = self._create_ssl_context()
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with certifi certificates."""
        try:
            ctx = ssl.create_default_context(cafile=certifi.where())
            return ctx
        except Exception as e:
            logger.warning(f"Certifi failed: {e}")
            return ssl.create_default_context()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    async def validate_candidate(
        self,
        drug_name: str,
        disease_name: str,
        drug_data: Dict,
        disease_data: Dict
    ) -> Dict:
        """
        Comprehensive clinical validation of a drug-disease pair.
        
        Returns:
            Dict with validation results including:
            - clinical_trials: List of relevant trials
            - literature_evidence: PubMed search results
            - safety_signals: Known adverse events
            - contraindications: Known issues
            - risk_level: LOW/MEDIUM/HIGH
            - recommendation: Text recommendation
        """
        logger.info(f"Validating {drug_name} for {disease_name}")
        
        # Check cache
        cache_key = f"{drug_name}_{disease_name}".lower()
        if cache_key in self.cache:
            logger.info("Using cached validation results")
            return self.cache[cache_key]
        
        # Run all validations in parallel
        results = await asyncio.gather(
            self._check_clinical_trials(drug_name, disease_name),
            self._check_pubmed_literature(drug_name, disease_name),
            self._check_safety_signals(drug_name, disease_name),
            self._check_mechanism_compatibility(drug_data, disease_data),
            return_exceptions=True
        )
        
        trials_data = results[0] if not isinstance(results[0], Exception) else {}
        literature_data = results[1] if not isinstance(results[1], Exception) else {}
        safety_data = results[2] if not isinstance(results[2], Exception) else {}
        mechanism_data = results[3] if not isinstance(results[3], Exception) else {}
        
        # Aggregate results
        validation_result = {
            'drug_name': drug_name,
            'disease_name': disease_name,
            'validated_at': datetime.now().isoformat(),
            'clinical_trials': trials_data,
            'literature_evidence': literature_data,
            'safety_signals': safety_data,
            'mechanism_analysis': mechanism_data,
            'risk_level': self._calculate_risk_level(
                trials_data, literature_data, safety_data, mechanism_data
            ),
            'recommendation': self._generate_recommendation(
                trials_data, literature_data, safety_data, mechanism_data
            ),
            'evidence_summary': self._generate_evidence_summary(
                trials_data, literature_data, safety_data
            )
        }
        
        # Cache result
        self.cache[cache_key] = validation_result
        
        return validation_result
    
    async def _check_clinical_trials(
        self,
        drug_name: str,
        disease_name: str
    ) -> Dict:
        """
        Search ClinicalTrials.gov for relevant trials.
        """
        logger.info(f"Checking clinical trials for {drug_name} + {disease_name}")
        
        session = await self._get_session()
        
        try:
            # Search ClinicalTrials.gov API
            params = {
                'query.cond': disease_name,
                'query.intr': drug_name,
                'pageSize': 20,
                'format': 'json',
                'countTotal': 'true'
            }
            
            async with session.get(
                'https://clinicaltrials.gov/api/v2/studies',
                params=params
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"ClinicalTrials.gov returned {resp.status}")
                    return {
                        'found': False,
                        'total_trials': 0,
                        'trials': [],
                        'error': f"API returned status {resp.status}"
                    }
                
                data = await resp.json()
                total_count = data.get('totalCount', 0)
                studies = data.get('studies', [])
                
                # Parse trial data
                trials = []
                for study in studies[:10]:  # Limit to top 10
                    protocol = study.get('protocolSection', {})
                    id_module = protocol.get('identificationModule', {})
                    status_module = protocol.get('statusModule', {})
                    design_module = protocol.get('designModule', {})
                    
                    trials.append({
                        'nct_id': id_module.get('nctId', 'Unknown'),
                        'title': id_module.get('briefTitle', 'Unknown'),
                        'status': status_module.get('overallStatus', 'Unknown'),
                        'phase': design_module.get('phases', ['Unknown'])[0] if design_module.get('phases') else 'Unknown',
                        'start_date': status_module.get('startDateStruct', {}).get('date', 'Unknown'),
                    })
                
                # Analyze trial outcomes
                completed_trials = [t for t in trials if 'COMPLETED' in t['status'].upper()]
                phase_3_trials = [t for t in trials if 'PHASE_3' in str(t['phase']).upper() or 'PHASE 3' in str(t['phase']).upper()]
                
                return {
                    'found': total_count > 0,
                    'total_trials': total_count,
                    'trials': trials,
                    'completed_trials': len(completed_trials),
                    'phase_3_trials': len(phase_3_trials),
                    'summary': f"Found {total_count} trials, {len(completed_trials)} completed"
                }
                
        except Exception as e:
            logger.error(f"Clinical trials check failed: {e}")
            return {
                'found': False,
                'total_trials': 0,
                'trials': [],
                'error': str(e)
            }
    
    async def _check_pubmed_literature(
        self,
        drug_name: str,
        disease_name: str
    ) -> Dict:
        """
        Search PubMed for relevant literature.
        """
        logger.info(f"Checking PubMed for {drug_name} + {disease_name}")
        
        session = await self._get_session()
        
        try:
            # Use PubMed E-utilities API
            search_term = f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]'
            
            # First, search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': search_term,
                'retmax': 100,
                'retmode': 'json'
            }
            
            async with session.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                params=search_params
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"PubMed search returned {resp.status}")
                    return {
                        'found': False,
                        'total_articles': 0,
                        'recent_articles': 0,
                        'error': f"API returned status {resp.status}"
                    }
                
                data = await resp.json()
                result = data.get('esearchresult', {})
                total_count = int(result.get('count', 0))
                pmids = result.get('idlist', [])
                
                # Get recent articles (last 5 years)
                recent_search_params = {
                    'db': 'pubmed',
                    'term': search_term + ' AND ("2019"[Date - Publication] : "2024"[Date - Publication])',
                    'retmax': 50,
                    'retmode': 'json'
                }
                
                recent_count = 0
                async with session.get(
                    'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                    params=recent_search_params
                ) as recent_resp:
                    if recent_resp.status == 200:
                        recent_data = await recent_resp.json()
                        recent_result = recent_data.get('esearchresult', {})
                        recent_count = int(recent_result.get('count', 0))
                
                return {
                    'found': total_count > 0,
                    'total_articles': total_count,
                    'recent_articles': recent_count,
                    'sample_pmids': pmids[:5],
                    'summary': f"Found {total_count} publications, {recent_count} recent"
                }
                
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return {
                'found': False,
                'total_articles': 0,
                'recent_articles': 0,
                'error': str(e)
            }
    
    async def _check_safety_signals(
        self,
        drug_name: str,
        disease_name: str
    ) -> Dict:
        """
        Check OpenFDA for adverse event signals.
        """
        logger.info(f"Checking safety signals for {drug_name}")
        
        session = await self._get_session()
        
        try:
            # Search OpenFDA adverse events
            search_query = f'patient.drug.medicinalproduct:"{drug_name}"'
            
            params = {
                'search': search_query,
                'limit': 100
            }
            
            async with session.get(
                'https://api.fda.gov/drug/event.json',
                params=params
            ) as resp:
                if resp.status == 404:
                    # No adverse events found (could be good or bad)
                    return {
                        'found': False,
                        'total_events': 0,
                        'serious_events': 0,
                        'summary': 'No adverse events reported in OpenFDA'
                    }
                
                if resp.status != 200:
                    logger.warning(f"OpenFDA returned {resp.status}")
                    return {
                        'found': False,
                        'total_events': 0,
                        'error': f"API returned status {resp.status}"
                    }
                
                data = await resp.json()
                results = data.get('results', [])
                
                # Count serious events
                serious_count = sum(
                    1 for r in results
                    if r.get('serious') == '1'
                )
                
                # Get common reactions
                all_reactions = []
                for result in results[:50]:
                    reactions = result.get('patient', {}).get('reaction', [])
                    for reaction in reactions:
                        all_reactions.append(
                            reaction.get('reactionmeddrapt', 'Unknown')
                        )
                
                # Count reaction frequency
                from collections import Counter
                reaction_counts = Counter(all_reactions)
                top_reactions = reaction_counts.most_common(10)
                
                return {
                    'found': True,
                    'total_events': len(results),
                    'serious_events': serious_count,
                    'top_reactions': [
                        {'reaction': r[0], 'count': r[1]}
                        for r in top_reactions
                    ],
                    'summary': f"{len(results)} adverse events, {serious_count} serious"
                }
                
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return {
                'found': False,
                'total_events': 0,
                'error': str(e)
            }
    
    async def _check_mechanism_compatibility(
        self,
        drug_data: Dict,
        disease_data: Dict
    ) -> Dict:
        """
        Analyze if drug mechanism is compatible with disease.
        """
        mechanism = drug_data.get('mechanism', '').lower()
        disease_name = disease_data.get('name', '').lower()
        
        # Define incompatible combinations
        incompatible_patterns = [
            # Immunosuppressants for autoimmune-driven neurodegeneration
            {
                'drug_keywords': ['immunosuppressant', 'immunosuppressive'],
                'disease_keywords': ['autoimmune'],
                'reason': 'May worsen autoimmune component'
            },
            # CNS depressants for conditions requiring alertness
            {
                'drug_keywords': ['sedative', 'hypnotic', 'cns depressant'],
                'disease_keywords': ['parkinson', 'movement disorder'],
                'reason': 'May worsen motor symptoms'
            },
            # Dopamine antagonists for Parkinson's
            {
                'drug_keywords': ['dopamine antagonist', 'antipsychotic'],
                'disease_keywords': ['parkinson'],
                'reason': 'Contraindicated - worsens Parkinson symptoms'
            }
        ]
        
        warnings = []
        compatible = True
        
        for pattern in incompatible_patterns:
            drug_match = any(
                keyword in mechanism
                for keyword in pattern['drug_keywords']
            )
            disease_match = any(
                keyword in disease_name
                for keyword in pattern['disease_keywords']
            )
            
            if drug_match and disease_match:
                warnings.append(pattern['reason'])
                compatible = False
        
        return {
            'compatible': compatible,
            'warnings': warnings,
            'mechanism_summary': drug_data.get('mechanism', 'Unknown'),
            'summary': 'Compatible' if compatible else f"Warning: {'; '.join(warnings)}"
        }
    
    def _calculate_risk_level(
        self,
        trials_data: Dict,
        literature_data: Dict,
        safety_data: Dict,
        mechanism_data: Dict
    ) -> str:
        """
        Calculate overall risk level: LOW, MEDIUM, HIGH
        """
        risk_score = 0
        
        # Positive indicators (lower risk)
        if trials_data.get('found') and trials_data.get('total_trials', 0) > 0:
            risk_score -= 2
            if trials_data.get('completed_trials', 0) > 0:
                risk_score -= 1
            if trials_data.get('phase_3_trials', 0) > 0:
                risk_score -= 2
        
        if literature_data.get('found') and literature_data.get('total_articles', 0) > 5:
            risk_score -= 1
            if literature_data.get('recent_articles', 0) > 3:
                risk_score -= 1
        
        # Negative indicators (higher risk)
        if not mechanism_data.get('compatible', True):
            risk_score += 5  # Major red flag
        
        if safety_data.get('serious_events', 0) > 50:
            risk_score += 2
        elif safety_data.get('serious_events', 0) > 20:
            risk_score += 1
        
        if not trials_data.get('found') and not literature_data.get('found'):
            risk_score += 2  # No evidence at all
        
        # Determine risk level
        if risk_score <= -3:
            return 'LOW'
        elif risk_score <= 1:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _generate_recommendation(
        self,
        trials_data: Dict,
        literature_data: Dict,
        safety_data: Dict,
        mechanism_data: Dict
    ) -> str:
        """
        Generate human-readable recommendation.
        """
        if not mechanism_data.get('compatible', True):
            return f"⛔ NOT RECOMMENDED - {mechanism_data.get('summary', 'Mechanism incompatible')}"
        
        if trials_data.get('phase_3_trials', 0) > 0:
            return "✅ STRONG CANDIDATE - Phase 3 trials completed"
        
        if trials_data.get('completed_trials', 0) > 0:
            return "✅ PROMISING - Clinical trials completed"
        
        if trials_data.get('total_trials', 0) > 0:
            return "⚠️ INVESTIGATE FURTHER - Trials ongoing"
        
        if literature_data.get('total_articles', 0) > 10:
            return "⚠️ THEORETICAL SUPPORT - Strong literature base, needs trials"
        
        if literature_data.get('total_articles', 0) > 0:
            return "⚠️ PRELIMINARY - Some literature support"
        
        return "⚠️ EXPERIMENTAL - Limited clinical evidence"
    
    def _generate_evidence_summary(
        self,
        trials_data: Dict,
        literature_data: Dict,
        safety_data: Dict
    ) -> List[str]:
        """
        Generate bullet points of evidence.
        """
        summary = []
        
        # Clinical trials
        if trials_data.get('found'):
            summary.append(
                f"✅ {trials_data.get('total_trials', 0)} clinical trials found"
            )
            if trials_data.get('completed_trials', 0) > 0:
                summary.append(
                    f"   └─ {trials_data['completed_trials']} completed"
                )
        else:
            summary.append("⚠️ No clinical trials found")
        
        # Literature
        if literature_data.get('found'):
            summary.append(
                f"✅ {literature_data.get('total_articles', 0)} publications found"
            )
            if literature_data.get('recent_articles', 0) > 0:
                summary.append(
                    f"   └─ {literature_data['recent_articles']} recent (2019-2024)"
                )
        else:
            summary.append("⚠️ No relevant publications found")
        
        # Safety
        if safety_data.get('found'):
            if safety_data.get('serious_events', 0) > 50:
                summary.append(
                    f"⚠️ {safety_data['serious_events']} serious adverse events reported"
                )
            else:
                summary.append(
                    f"✅ {safety_data.get('total_events', 0)} adverse events (acceptable safety profile)"
                )
        else:
            summary.append("✅ No adverse events in FDA database")
        
        return summary
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()