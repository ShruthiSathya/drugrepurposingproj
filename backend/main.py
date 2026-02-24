from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from pipeline.production_pipeline import ProductionPipeline
from pipeline.clinical_validator import ClinicalValidator
from pipeline.drug_filter import DrugSafetyFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Drug Repurposing API",
    description="AI-powered drug repurposing using gene-disease relationships",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    global pipeline
    logger.info("🚀 Starting Drug Repurposing API...")
    logger.info("📊 Databases: OpenTargets, ChEMBL, DGIdb, ClinicalTrials.gov")
    try:
        pipeline = ProductionPipeline()
        # ProductionPipeline initializes itself in __init__, no separate initialize() needed
        logger.info("✅ API ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global pipeline
    # ProductionPipeline doesn't have a close() method
    logger.info("👋 API shutdown complete")

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Drug Repurposing API",
        "version": "2.0.0"
    }

@app.post("/analyze", tags=["Analysis"])
async def analyze_disease(request: dict):
    """
    Analyze a disease and find repurposing candidates with safety filtering.
    """
    global pipeline
    
    if not pipeline:
        return {
            "success": False,
            "error": "Pipeline not initialized"
        }
    
    try:
        disease_name = request.get('disease_name')
        min_score = request.get('min_score', 0.2)
        max_results = request.get('max_results', 10)
        
        if not disease_name:
            return {
                "success": False,
                "error": "Missing disease_name"
            }
        
        logger.info(f"Analysis request: {disease_name}")
        
        # Run gene-based analysis
        result = await pipeline.analyze_disease(
            disease_name=disease_name,
            min_score=min_score,
            max_results=max_results * 2  # Get extra candidates before filtering
        )
        
        if not result['success']:
            return result
        
        # ⭐ FIX: Ensure candidates have the required fields for filtering
        candidates = result.get('candidates', [])
        for candidate in candidates:
            # Ensure 'indication' field exists (drug_filter expects 'indication')
            if 'indication' not in candidate and 'original_indication' in candidate:
                candidate['indication'] = candidate['original_indication']
            elif 'indication' not in candidate:
                candidate['indication'] = ''
            
            # Ensure 'mechanism' field exists
            if 'mechanism' not in candidate:
                candidate['mechanism'] = ''
        
        # ⭐ FIXED: Apply safety filter with CORRECT settings
        safety_filter = DrugSafetyFilter()
        
        original_count = len(candidates)
        
        try:
            # FIXED: Set remove_relative=True to filter out ALL contraindicated drugs
            # This is critical for safety - "relative" contraindications like olanzapine
            # for diabetes or beta-blockers for asthma are still dangerous!
            safe_candidates, filtered_out = await safety_filter.filter_candidates(
                candidates=candidates,
                disease_name=disease_name,
                remove_absolute=True,   # Remove absolutely contraindicated
                remove_relative=True    # FIXED: Also remove relatively contraindicated (was False!)
            )
            
            # Limit to requested max_results after filtering
            safe_candidates = safe_candidates[:max_results]
            
            logger.info(
                f"Safety filter: {original_count} → {len(safe_candidates)} candidates "
                f"({len(filtered_out)} filtered out)"
            )
            
            # Update result with filtered candidates
            result['candidates'] = safe_candidates
            result['filtered_count'] = len(filtered_out)
            result['filtered_drugs'] = [
                {
                    'drug_name': c['drug_name'],
                    'reason': c.get('contraindication', {}).get('reason', 'Unknown'),
                    'severity': c.get('contraindication', {}).get('severity', 'unknown')
                }
                for c in filtered_out
            ]
            
        except Exception as filter_error:
            logger.error(f"Safety filter error: {filter_error}")
            # If filtering fails, return unfiltered results with warning
            result['candidates'] = candidates[:max_results]
            result['filtered_count'] = 0
            result['filtered_drugs'] = []
            result['filter_warning'] = f"Safety filter error: {str(filter_error)}"
        
        return result
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/validate_clinical", tags=["Analysis"])
async def validate_clinical(request: dict):
    """
    Validate a drug candidate clinically using multiple databases.
    
    Checks:
    - Clinical trials (ClinicalTrials.gov)
    - Literature evidence (PubMed)
    - Safety signals (OpenFDA)
    - Mechanism compatibility
    """
    global pipeline
    
    if not pipeline:
        return {
            "success": False,
            "error": "Pipeline not initialized"
        }
    
    try:
        drug_name = request.get('drug_name')
        disease_name = request.get('disease_name')
        drug_data = request.get('drug_data', {})
        disease_data = request.get('disease_data', {})
        
        if not drug_name or not disease_name:
            return {
                "success": False,
                "error": "Missing drug_name or disease_name"
            }
        
        logger.info(f"Clinical validation request: {drug_name} for {disease_name}")
        
        # Create validator
        validator = ClinicalValidator()
        
        try:
            # Run validation
            validation_result = await validator.validate_candidate(
                drug_name=drug_name,
                disease_name=disease_name,
                drug_data=drug_data,
                disease_data=disease_data
            )
            
            return {
                "success": True,
                "validation": validation_result
            }
        
        finally:
            await validator.close()
    
    except Exception as e:
        logger.error(f"Clinical validation error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e)
        }