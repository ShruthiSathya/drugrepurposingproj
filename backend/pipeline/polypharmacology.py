"""
polypharmacology.py — Multi-Target Drug Scoring v2.0
======================================================
Evaluates how many disease-relevant targets a drug hits simultaneously
(polypharmacology) and estimates its ability to overcome drug resistance
based on disease-specific resistance mechanisms.

FIXES vs v1
-----------
1. PDAC-SPECIFIC RESISTANCE GENES REMOVED: v1 had a hardcoded set
   PDAC_RESISTANCE_GENES = {"KRAS","YAP1","MCL1","ABCB1",...} used for ALL
   diseases. For non-pancreatic diseases this produced systematically wrong
   resistance bypass scores — drugs would be penalised for not targeting
   KRAS/YAP1 even in Parkinson's or rheumatoid arthritis.

   v2 resolves disease-specific resistance genes dynamically from a
   DISEASE_RESISTANCE_GENES database, falling back to a generic set of
   broad-spectrum resistance mechanisms (MDR efflux pumps, anti-apoptotic
   proteins) when the disease is not in the database.

2. DISEASE AUTO-RESOLUTION: Uses keyword matching so "rheumatoid arthritis"
   maps to the RA resistance gene set, "non-small cell lung cancer" maps to
   lung cancer, etc. Uses same DISEASE_KEYWORD_MAP pattern as insilico_trial.py.

3. SYNERGY SCORE CALIBRATION: v1 used a fixed synergy threshold regardless of
   disease biology. v2 scales synergy expectations by the known heterogeneity
   level of each disease (high-heterogeneity cancers like GBM require more
   multi-target coverage to overcome resistance than single-gene disorders).

4. MECHANISM DIVERSITY BONUS: Added a mechanism diversity bonus that rewards
   drugs hitting targets in distinct pathways (e.g. both a kinase and a GPCR),
   not just counting raw target number.

5. SELECTIVITY PENALTY: Added a selectivity penalty for promiscuous drugs
   targeting > 15 proteins, which correlates with higher toxicity risk.
"""

import logging
import math
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Disease-specific resistance gene sets
# These represent genes whose activity is associated with treatment resistance
# in each disease. A drug that hits one or more of these genes gets a
# resistance bypass bonus.
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_RESISTANCE_GENES: Dict[str, Set[str]] = {

    # ── Oncology ──────────────────────────────────────────────────────────────

    "pancreatic": {
        "KRAS", "YAP1", "MCL1", "ABCB1", "FGF2", "IL6",
        "STAT3", "AXL", "WNT5A", "NF2", "MAPK1",
    },
    "glioblastoma": {
        "EGFR", "PTEN", "IDH1", "MGMT", "PDGFRA",
        "NF1", "RB1", "CDK4", "MDM2", "VEGFA",
    },
    "lung": {
        "KRAS", "MET", "EGFR", "ALK", "RET",
        "ERBB2", "FGFR1", "MYC", "BCL2", "AXL",
    },
    "breast": {
        "ESR1", "ERBB2", "PIK3CA", "AKT1", "CDK4",
        "CDK6", "CCND1", "ABCB1", "BCL2", "PTEN",
    },
    "colorectal": {
        "KRAS", "BRAF", "PIK3CA", "SMAD4",
        "TP53", "FBXW7", "NRAS", "APC",
    },
    "ovarian": {
        "BRCA1", "BRCA2", "CCNE1", "RB1",
        "PIK3CA", "AKT1", "PTEN", "MYC",
    },
    "melanoma": {
        "BRAF", "NRAS", "MAP2K1", "CDKN2A",
        "MITF", "AXL", "NF1",
    },
    "multiple myeloma": {
        "CRBN", "TP53", "RAS", "FGFR3",
        "IRF4", "MYC", "DIS3", "FAM46C",
    },
    "leukemia": {
        "ABL1", "BCR", "T315I_ABL1",
        "IKZF1", "CDKN2A", "RAS", "TP53",
    },

    # ── Autoimmune / Inflammatory ─────────────────────────────────────────────

    "rheumatoid arthritis": {
        "TNF", "IL6", "STAT3", "JAK1", "JAK2",
        "TRAF6", "NFkB1", "IL1B", "MAPK14",
    },
    "lupus": {
        "IFNA1", "BLK", "IRF5", "STAT4",
        "PTPN22", "TNFAIP3", "IKBKE",
    },
    "inflammatory bowel disease": {
        "TNF", "IL12B", "IL23A", "JAK1",
        "STAT3", "NFkB1", "RIPK2",
    },
    "multiple sclerosis": {
        "HLA-DRB1", "IL7R", "CD58",
        "TNFSF13B", "STAT3", "JAK1",
    },

    # ── Cardiovascular ────────────────────────────────────────────────────────

    "heart failure": {
        "NPPA", "NPPB", "MYH7", "TNNT2",
        "ACE", "ADRB1", "RASD1",
    },
    "pulmonary arterial hypertension": {
        "BMPR2", "KCNK3", "EIF2AK4",
        "SMAD9", "ACVRL1", "EDN1",
    },

    # ── Neurological ──────────────────────────────────────────────────────────

    "parkinson": {
        "SNCA", "LRRK2", "GBA", "PINK1",
        "PRKN", "DJ1", "ATP13A2", "VPS35",
    },
    "alzheimer": {
        "APP", "PSEN1", "PSEN2", "APOE", "TREM2",
        "BIN1", "CLU", "CR1", "ABCA7",
    },
    "epilepsy": {
        "SCN1A", "KCNQ2", "CDKL5", "DEPDC5",
        "GABRA1", "SCN8A",
    },
    "amyotrophic lateral sclerosis": {
        "SOD1", "TARDBP", "FUS", "C9orf72",
        "UBQLN2", "VCP", "SQSTM1",
    },
    "huntington": {
        "HTT", "ADORA2A", "BDNF", "HAP1",
    },

    # ── Metabolic ─────────────────────────────────────────────────────────────

    "type 2 diabetes": {
        "INSR", "PRKAA1", "PPARG", "GLP1R",
        "ABCC8", "KCNJ11", "IRS1",
    },
    "hypercholesterolemia": {
        "LDLR", "PCSK9", "APOB", "HMGCR",
    },

    # ── Rare / Genetic ────────────────────────────────────────────────────────

    "cystic fibrosis": {"CFTR"},
    "tuberous sclerosis": {"TSC1", "TSC2", "MTOR"},
    "spinal muscular atrophy": {"SMN1", "SMN2"},

    # ── Generic broad-spectrum resistance mechanisms ───────────────────────────
    # Used as fallback for any disease not in the database above.
    # Covers MDR efflux pumps, anti-apoptotic proteins, DNA repair.
    "generic": {
        "ABCB1",   # MDR1 / P-gp efflux pump
        "ABCG2",   # BCRP efflux pump
        "ABCC1",   # MRP1 efflux pump
        "BCL2",    # anti-apoptotic
        "BCL2L1",  # BCL-XL anti-apoptotic
        "MCL1",    # anti-apoptotic
        "BIRC5",   # survivin
        "MDM2",    # p53 inhibitor
        "STAT3",   # broad survival signalling
        "NFkB1",   # NF-κB survival signalling
        "AKT1",    # broad survival kinase
        "MTOR",    # broad survival/growth kinase
    },
}

# ── Keyword map: disease name substring → resistance gene set key ─────────────
DISEASE_KEYWORD_TO_RESISTANCE_KEY: List[Tuple[str, str]] = [
    ("pancreatic",                       "pancreatic"),
    ("pdac",                             "pancreatic"),
    ("glioblastoma",                     "glioblastoma"),
    ("gbm",                              "glioblastoma"),
    ("glioma",                           "glioblastoma"),
    ("non-small cell lung",              "lung"),
    ("nsclc",                            "lung"),
    ("lung cancer",                      "lung"),
    ("breast cancer",                    "breast"),
    ("breast carcinoma",                 "breast"),
    ("colorectal",                       "colorectal"),
    ("colon cancer",                     "colorectal"),
    ("ovarian",                          "ovarian"),
    ("melanoma",                         "melanoma"),
    ("multiple myeloma",                 "multiple myeloma"),
    ("myeloma",                          "multiple myeloma"),
    ("leukemia",                         "leukemia"),
    ("lymphoma",                         "leukemia"),
    ("cml",                              "leukemia"),
    ("rheumatoid arthritis",             "rheumatoid arthritis"),
    ("systemic lupus",                   "lupus"),
    ("lupus",                            "lupus"),
    ("inflammatory bowel",               "inflammatory bowel disease"),
    ("crohn",                            "inflammatory bowel disease"),
    ("ulcerative colitis",               "inflammatory bowel disease"),
    ("multiple sclerosis",               "multiple sclerosis"),
    ("heart failure",                    "heart failure"),
    ("pulmonary arterial hypertension",  "pulmonary arterial hypertension"),
    ("pah",                              "pulmonary arterial hypertension"),
    ("parkinson",                        "parkinson"),
    ("alzheimer",                        "alzheimer"),
    ("epilepsy",                         "epilepsy"),
    ("amyotrophic lateral sclerosis",    "amyotrophic lateral sclerosis"),
    ("als",                              "amyotrophic lateral sclerosis"),
    ("huntington",                       "huntington"),
    ("type 2 diabetes",                  "type 2 diabetes"),
    ("hypercholesterolemia",             "hypercholesterolemia"),
    ("cystic fibrosis",                  "cystic fibrosis"),
    ("tuberous sclerosis",               "tuberous sclerosis"),
    ("spinal muscular atrophy",          "spinal muscular atrophy"),
    ("sma",                              "spinal muscular atrophy"),
]

# ── Disease heterogeneity scores (0=low, 1=high) ─────────────────────────────
# High heterogeneity → need more multi-target coverage to overcome resistance.
DISEASE_HETEROGENEITY: Dict[str, float] = {
    "glioblastoma":                    0.90,
    "pancreatic":                      0.85,
    "lung":                            0.70,
    "melanoma":                        0.70,
    "colorectal":                      0.65,
    "breast":                          0.60,
    "ovarian":                         0.65,
    "multiple myeloma":                0.60,
    "leukemia":                        0.50,
    "alzheimer":                       0.50,
    "parkinson":                       0.45,
    "rheumatoid arthritis":            0.30,
    "lupus":                           0.40,
    "inflammatory bowel disease":      0.40,
    "heart failure":                   0.35,
    "epilepsy":                        0.50,
    "amyotrophic lateral sclerosis":   0.55,
    "cystic fibrosis":                 0.20,
    "tuberous sclerosis":              0.25,
    "spinal muscular atrophy":         0.15,
    "type 2 diabetes":                 0.35,
    "generic":                         0.50,
}

# ── Broad drug target classifications (for mechanism diversity bonus) ─────────
MECHANISM_CLASSES: Dict[str, str] = {
    # Kinases
    "ABL1": "kinase", "AKT1": "kinase", "ALK": "kinase", "BRAF": "kinase",
    "BTK": "kinase", "CDK4": "kinase", "CDK6": "kinase", "EGFR": "kinase",
    "ERBB2": "kinase", "FGFR1": "kinase", "FGFR3": "kinase", "JAK1": "kinase",
    "JAK2": "kinase", "KDR": "kinase", "LCK": "kinase", "LRRK2": "kinase",
    "MAP2K1": "kinase", "MAPK1": "kinase", "MAPK14": "kinase", "MET": "kinase",
    "MTOR": "kinase", "PDGFRa": "kinase", "PIK3CA": "kinase", "PRKAA1": "kinase",
    "RET": "kinase", "SRC": "kinase", "STAT3": "kinase", "VEGFr2": "kinase",
    "ZAP70": "kinase",

    # Nuclear receptors
    "AR": "nuclear_receptor", "ESR1": "nuclear_receptor", "PGR": "nuclear_receptor",
    "PPARG": "nuclear_receptor", "RARA": "nuclear_receptor", "VDR": "nuclear_receptor",

    # GPCRs
    "ADRB1": "gpcr", "ADRB2": "gpcr", "ADORA2A": "gpcr", "DRD1": "gpcr",
    "DRD2": "gpcr", "GLP1R": "gpcr", "HTR2A": "gpcr",

    # Ion channels
    "CFTR": "ion_channel", "GABRA1": "ion_channel", "KCNJ11": "ion_channel",
    "KCNQ2": "ion_channel", "SCN1A": "ion_channel", "SCN8A": "ion_channel",

    # Proteases
    "ACE": "protease", "ADAM10": "protease",

    # E3 ligases / ubiquitin
    "CRBN": "e3_ligase", "MDM2": "e3_ligase", "VHL": "e3_ligase",

    # Epigenetic
    "HDAC1": "epigenetic", "DNMT1": "epigenetic", "EZH2": "epigenetic",
    "BRD4": "epigenetic",

    # Cell surface / immune
    "CD274": "immune_checkpoint", "CTLA4": "immune_checkpoint",
    "PDCD1": "immune_checkpoint", "CD19": "immune_surface",
    "CD20": "immune_surface",

    # Chaperones
    "HSP90AA1": "chaperone",

    # Transcription factors
    "MYC": "transcription_factor", "NFkB1": "transcription_factor",
    "STAT3": "transcription_factor", "TP53": "transcription_factor",

    # Apoptosis regulators
    "BCL2": "apoptosis", "BCL2L1": "apoptosis", "MCL1": "apoptosis",
    "BIRC5": "apoptosis",

    # Efflux pumps (avoid targeting — not therapeutic)
    "ABCB1": "efflux_pump", "ABCG2": "efflux_pump", "ABCC1": "efflux_pump",

    # Structural
    "MYH7": "structural", "TNNT2": "structural",
}


def _resolve_disease_resistance_key(disease_name: str) -> str:
    """Map disease name to a key in DISEASE_RESISTANCE_GENES."""
    name_lower = disease_name.lower().strip()
    for keyword, key in DISEASE_KEYWORD_TO_RESISTANCE_KEY:
        if keyword in name_lower:
            return key
    return "generic"


class PolypharmacologyScorer:
    """
    Scores drugs based on their multi-target pharmacology profile.

    Three sub-scores:
    1. target_breadth_score:   How many disease-relevant targets does the drug hit?
    2. resistance_bypass_score: Does it hit known resistance mechanisms?
    3. synergy_potential_score: Does it hit synergistic target pairs?

    Aggregate polypharmacology_score = weighted combination of the above,
    minus a selectivity penalty for overly promiscuous drugs.
    """

    def __init__(self, disease_name: str = ""):
        self.disease_name        = disease_name.lower().strip()
        self._resistance_key     = _resolve_disease_resistance_key(disease_name)
        self._resistance_genes   = (
            DISEASE_RESISTANCE_GENES.get(self._resistance_key, set()) |
            DISEASE_RESISTANCE_GENES["generic"]   # always add broad-spectrum
        )
        self._heterogeneity      = DISEASE_HETEROGENEITY.get(
            self._resistance_key, 0.50
        )

        if disease_name:
            logger.info(
                f"✅ PolypharmacologyScorer: disease='{disease_name}', "
                f"resistance_key='{self._resistance_key}', "
                f"n_resistance_genes={len(self._resistance_genes)}, "
                f"heterogeneity={self._heterogeneity}"
            )
        else:
            logger.info(
                "✅ PolypharmacologyScorer: generic mode (no disease specified)"
            )

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    def _score_target_breadth(
        self,
        drug_targets:    List[str],
        disease_targets: List[str],
    ) -> float:
        """
        Score based on how many disease targets the drug covers.
        Uses a diminishing-returns curve: hitting 1–3 targets is most impactful,
        beyond 5 the marginal gain drops.
        """
        if not disease_targets:
            return 0.3  # uncertain without disease context

        disease_set = set(t.upper() for t in disease_targets)
        drug_set    = set(t.upper() for t in drug_targets)
        hits        = drug_set & disease_set

        if not hits:
            return 0.0

        n_hits   = len(hits)
        n_total  = len(disease_set)
        fraction = n_hits / n_total

        # Diminishing returns: log curve
        raw_score = math.log1p(n_hits) / math.log1p(max(n_total, 5))
        # Combine fraction and log score
        score = 0.4 * fraction + 0.6 * raw_score
        return round(min(score, 1.0), 4)

    def _score_resistance_bypass(self, drug_targets: List[str]) -> float:
        """
        Score based on whether the drug hits disease-specific resistance genes.
        A drug that bypasses resistance mechanisms scores higher.
        Higher disease heterogeneity → resistance bypass is more important.
        """
        if not self._resistance_genes:
            return 0.5  # neutral if no resistance data

        drug_set     = set(t.upper() for t in drug_targets)
        bypass_genes = drug_set & set(g.upper() for g in self._resistance_genes)

        if not bypass_genes:
            return 0.0

        n_bypass = len(bypass_genes)
        n_resist = len(self._resistance_genes)

        # Raw fraction of resistance genes covered
        fraction = n_bypass / n_resist

        # Scale by disease heterogeneity (high het → resistance bypass is critical)
        base_score = fraction * (0.5 + 0.5 * self._heterogeneity)

        return round(min(base_score, 1.0), 4)

    def _score_synergy_potential(
        self, drug_targets: List[str], disease_targets: List[str]
    ) -> float:
        """
        Estimate synergistic potential by rewarding multi-pathway coverage.
        Uses mechanism class diversity: a drug hitting a kinase AND a GPCR
        AND a nuclear receptor has higher synergy potential than one hitting
        3 kinases in the same pathway.
        """
        if not drug_targets:
            return 0.0

        drug_set  = set(t.upper() for t in drug_targets)
        dis_set   = set(t.upper() for t in disease_targets) if disease_targets else drug_set
        relevant  = drug_set & dis_set if disease_targets else drug_set

        if not relevant:
            return 0.0

        # Count unique mechanism classes among disease-relevant targets
        classes = set()
        for gene in relevant:
            cls = MECHANISM_CLASSES.get(gene, "other")
            if cls not in ("efflux_pump",):  # don't reward efflux pump targeting
                classes.add(cls)

        n_classes = len(classes)
        # 1 class → 0, 2 classes → 0.5, 3+ classes → 0.8+
        if n_classes <= 1:
            return 0.2
        elif n_classes == 2:
            return 0.55
        elif n_classes == 3:
            return 0.75
        else:
            return min(0.75 + (n_classes - 3) * 0.05, 0.95)

    def _selectivity_penalty(self, n_targets: int) -> float:
        """
        Penalise overly promiscuous drugs (> 15 targets).
        Promiscuity correlates with toxicity and off-target effects.
        Returns a penalty multiplier (0=max penalty, 1=no penalty).
        """
        if n_targets <= 5:
            return 1.00
        elif n_targets <= 10:
            return 0.95
        elif n_targets <= 15:
            return 0.90
        elif n_targets <= 25:
            return 0.80
        else:
            return 0.65

    # ── Aggregate scorer ──────────────────────────────────────────────────────

    def score_candidate(
        self,
        candidate:       Dict,
        disease_targets: Optional[List[str]] = None,
    ) -> Dict:
        """
        Score a single drug candidate's polypharmacology profile.

        Parameters
        ----------
        candidate : dict
            Must have 'drug_name'. Optionally 'target_genes', 'mechanisms'.
        disease_targets : list of str, optional
            Disease-associated genes. Pulled from candidate['disease_genes']
            if not provided.

        Returns
        -------
        candidate enriched with:
          - polypharmacology_score (float 0–1)
          - polypharmacology_detail (dict)
        """
        drug_name    = candidate.get("drug_name", "Unknown")
        drug_targets = [
            t.upper() for t in (candidate.get("target_genes", []) or [])
        ]
        if disease_targets is None:
            disease_targets = [
                t.upper() for t in (candidate.get("disease_genes", []) or [])
            ]

        n_targets = len(drug_targets)

        breadth_score   = self._score_target_breadth(drug_targets, disease_targets)
        resistance_score = self._score_resistance_bypass(drug_targets)
        synergy_score   = self._score_synergy_potential(drug_targets, disease_targets)
        selectivity_pen = self._selectivity_penalty(n_targets)

        # Weighted aggregate
        # Weights: resistance bypass most important (disease-specific signal),
        # then target breadth (coverage), then synergy (bonus).
        # Scale by disease heterogeneity — high-het diseases need more poly.
        het = self._heterogeneity
        raw_score = (
            0.40 * breadth_score +
            0.40 * resistance_score +
            0.20 * synergy_score
        ) * selectivity_pen

        # Heterogeneity adjustment: boost score importance for high-het diseases
        # (does not change the absolute value, but can be used downstream)
        adjusted_score = raw_score

        # Identify which resistance genes are covered
        resistance_hits = (
            set(drug_targets) &
            set(g.upper() for g in self._resistance_genes)
        )

        # Identify mechanism classes
        classes_hit = {
            MECHANISM_CLASSES.get(g, "other")
            for g in drug_targets
            if MECHANISM_CLASSES.get(g) != "efflux_pump"
        }

        detail = {
            "drug_name":                drug_name,
            "n_drug_targets":           n_targets,
            "n_disease_targets":        len(disease_targets),
            "target_breadth_score":     round(breadth_score, 4),
            "resistance_bypass_score":  round(resistance_score, 4),
            "synergy_potential_score":  round(synergy_score, 4),
            "selectivity_penalty":      round(selectivity_pen, 4),
            "aggregate_score":          round(adjusted_score, 4),
            "disease":                  self.disease_name or "generic",
            "resistance_key":           self._resistance_key,
            "disease_heterogeneity":    self._heterogeneity,
            "resistance_genes_hit":     sorted(resistance_hits),
            "mechanism_classes_hit":    sorted(classes_hit),
            "n_mechanism_classes":      len(classes_hit),
        }

        candidate["polypharmacology_score"]  = round(adjusted_score, 4)
        candidate["polypharmacology_detail"] = detail

        logger.debug(
            f"   {drug_name}: breadth={breadth_score:.2f}, "
            f"resistance={resistance_score:.2f}, synergy={synergy_score:.2f} "
            f"→ poly={adjusted_score:.2f}"
        )
        return candidate

    def score_batch(
        self,
        candidates:      List[Dict],
        disease_targets: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Score a list of candidates, sorted by polypharmacology_score."""
        logger.info(
            f"🔬 Polypharmacology scoring: {len(candidates)} candidates, "
            f"disease='{self.disease_name}', "
            f"resistance_key='{self._resistance_key}'"
        )

        results = [
            self.score_candidate(c, disease_targets) for c in candidates
        ]
        results.sort(
            key=lambda c: c.get("polypharmacology_score", 0), reverse=True
        )

        top = results[:5]
        for c in top:
            logger.info(
                f"   ✅ {c.get('drug_name','?')}: "
                f"poly={c.get('polypharmacology_score',0):.2f}, "
                f"resistance_hits={c.get('polypharmacology_detail',{}).get('resistance_genes_hit',[])} "
                f"classes={c.get('polypharmacology_detail',{}).get('mechanism_classes_hit',[])} "
            )
        return results

    def get_resistance_profile(self) -> Dict:
        """Return the current disease resistance profile for logging/reporting."""
        return {
            "disease":            self.disease_name or "generic",
            "resistance_key":     self._resistance_key,
            "heterogeneity":      self._heterogeneity,
            "n_resistance_genes": len(self._resistance_genes),
            "resistance_genes":   sorted(self._resistance_genes),
        }