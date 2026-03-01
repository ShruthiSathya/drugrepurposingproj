"""
repodb_benchmark.py — RepoDB Benchmarking Script
=================================================
Evaluates the drug repurposing pipeline against the RepoDB gold-standard
dataset of approved drug-indication pairs.


RepoDB Reference
----------------
Brown AS & Patel CJ (2017). "A standard database for drug repositioning."
Pac Symp Biocomput, 22, 393-401. PMID:27896997.
Dataset: https://unmtid-shinyapps.net/shiny/repodb/

Metrics computed
----------------
  AUC-ROC  — area under ROC curve (primary benchmark metric)
  AUC-PR   — area under Precision-Recall curve
  Hit@N    — fraction of RepoDB positives in top-N candidates per disease
  MRR      — mean reciprocal rank of known positives

Usage
-----
    python repodb_benchmark.py [--repodb-file repodb.csv] [--top-n 50]
                               [--output repodb_benchmark_results.json]
                               [--max-diseases 20]
"""

import argparse
import asyncio
import csv
import json
import logging
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.pipeline.production_pipeline import ProductionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPODB_PRIMARY_URL  = "https://raw.githubusercontent.com/emckiernan/repoDB/master/repodb.csv"
_REPODB_FALLBACK_URL = "https://raw.githubusercontent.com/rhenanbartels/repodb/master/data-raw/repodb.csv"

# ─────────────────────────────────────────────────────────────────────────────
# FIX 6a: RepoDB → OpenTargets EFO name map
#
# Keys are RepoDB ind_name values (lowercased).
# Values are OpenTargets disease name strings that fetch_disease_data() can resolve.
#
# Source: manually verified against OpenTargets Platform EFO ontology.
# Expand this map as you run more diseases and find mismatches.
# ─────────────────────────────────────────────────────────────────────────────
REPODB_TO_EFO: Dict[str, str] = {
    # Infectious disease
    "aids with kaposi's sarcoma":           "Kaposi sarcoma",
    "aids-related kaposi's sarcoma":        "Kaposi sarcoma",
    "abdominal abscess":                    "bacterial infectious disease",
    "abdominal infection":                  "bacterial infectious disease",
    "acinetobacter infections":             "Acinetobacter infection",
    "acanthamoeba keratitis":               "Acanthamoeba keratitis",
    "hiv infection":                        "human immunodeficiency virus infectious disease",
    "hiv-1 infection":                      "human immunodeficiency virus infectious disease",
    "hiv infections":                       "human immunodeficiency virus infectious disease",
    "bacterial infection":                  "bacterial infectious disease",
    "bacterial infections":                 "bacterial infectious disease",
    "fungal infection":                     "fungal infectious disease",
    "candidiasis":                          "candidiasis",
    "aspergillosis":                        "aspergillosis",
    "pneumonia":                            "pneumonia",
    "sepsis":                               "sepsis",
    "urinary tract infection":              "urinary tract infectious disease",
    "urinary tract infections":             "urinary tract infectious disease",

    # Oncology
    "acute lymphoblastic leukemia":         "acute lymphoblastic leukemia",
    "acute myeloid leukemia":               "acute myeloid leukemia",
    "chronic myeloid leukemia":             "chronic myelogenous leukemia",
    "chronic lymphocytic leukemia":         "chronic lymphocytic leukemia",
    "non-hodgkin lymphoma":                 "non-Hodgkin lymphoma",
    "hodgkin lymphoma":                     "Hodgkin lymphoma",
    "hodgkin's disease":                    "Hodgkin lymphoma",
    "multiple myeloma":                     "multiple myeloma",
    "breast cancer":                        "breast cancer",
    "breast neoplasms":                     "breast cancer",
    "lung cancer":                          "lung carcinoma",
    "non-small cell lung carcinoma":        "non-small cell lung carcinoma",
    "small cell lung carcinoma":            "small cell lung carcinoma",
    "colorectal cancer":                    "colorectal cancer",
    "colon cancer":                         "colon carcinoma",
    "rectal cancer":                        "rectal carcinoma",
    "ovarian cancer":                       "ovarian carcinoma",
    "prostate cancer":                      "prostate carcinoma",
    "melanoma":                             "melanoma",
    "glioblastoma":                         "glioblastoma multiforme",
    "renal cell carcinoma":                 "renal carcinoma",
    "bladder cancer":                       "bladder carcinoma",
    "pancreatic cancer":                    "pancreatic carcinoma",
    "gastric cancer":                       "stomach carcinoma",
    "stomach cancer":                       "stomach carcinoma",
    "hepatocellular carcinoma":             "hepatocellular carcinoma",
    "thyroid cancer":                       "thyroid gland carcinoma",
    "endometrial cancer":                   "endometrial carcinoma",
    "cervical cancer":                      "cervical carcinoma",
    "head and neck cancer":                 "head and neck carcinoma",
    "squamous cell carcinoma":              "squamous cell carcinoma",

    # Cardiovascular
    "hypertension":                         "hypertension",
    "heart failure":                        "heart failure",
    "coronary artery disease":              "coronary artery disease",
    "myocardial infarction":                "myocardial infarction",
    "acute st segment elevation myocardial infarction (disorder)": "myocardial infarction",
    "acute myocardial infarction":          "myocardial infarction",
    "st elevation myocardial infarction":   "myocardial infarction",
    "atrial fibrillation":                  "atrial fibrillation",
    "angina pectoris":                      "angina pectoris",
    "pulmonary arterial hypertension":      "pulmonary arterial hypertension",
    "pulmonary hypertension":               "pulmonary hypertension",
    "deep vein thrombosis":                 "deep vein thrombosis",
    "stroke":                               "stroke",
    "peripheral vascular disease":          "peripheral vascular disease",
    "hyperlipidemia":                       "hypercholesterolemia",
    "hypercholesterolemia":                 "hypercholesterolemia",
    "dyslipidemia":                         "hypercholesterolemia",

    # Neurological
    "alzheimer's disease":                  "Alzheimer disease",
    "alzheimer disease":                    "Alzheimer disease",
    "parkinson's disease":                  "Parkinson disease",
    "parkinson disease":                    "Parkinson disease",
    "multiple sclerosis":                   "multiple sclerosis",
    "epilepsy":                             "epilepsy",
    "absence epilepsy":                     "absence epilepsy",
    "migraine":                             "migraine",
    "schizophrenia":                        "schizophrenia",
    "bipolar disorder":                     "bipolar disorder",
    "major depressive disorder":            "major depressive disorder",
    "depression":                           "major depressive disorder",
    "anxiety disorder":                     "anxiety disorder",
    "amyotrophic lateral sclerosis":        "amyotrophic lateral sclerosis",
    "huntington's disease":                 "Huntington disease",
    "neuropathic pain":                     "neuropathic pain",
    "attention deficit hyperactivity disorder": "attention deficit hyperactivity disorder",
    "adhd":                                 "attention deficit hyperactivity disorder",

    # Metabolic / endocrine
    "type 2 diabetes":                      "type 2 diabetes mellitus",
    "type 2 diabetes mellitus":             "type 2 diabetes mellitus",
    "type 1 diabetes":                      "type 1 diabetes mellitus",
    "type 1 diabetes mellitus":             "type 1 diabetes mellitus",
    "obesity":                              "obesity",
    "gout":                                 "gout",
    "osteoporosis":                         "osteoporosis",
    "hypothyroidism":                       "hypothyroidism",
    "hyperthyroidism":                      "hyperthyroidism",
    "cushing's syndrome":                   "Cushing syndrome",
    "polycystic ovary syndrome":            "polycystic ovary syndrome",

    # Autoimmune / inflammatory
    "rheumatoid arthritis":                 "rheumatoid arthritis",
    "systemic lupus erythematosus":         "systemic lupus erythematosus",
    "lupus":                                "systemic lupus erythematosus",
    "inflammatory bowel disease":           "inflammatory bowel disease",
    "crohn's disease":                      "Crohn disease",
    "ulcerative colitis":                   "ulcerative colitis",
    "psoriasis":                            "psoriasis",
    "ankylosing spondylitis":               "ankylosing spondylitis",
    "pericarditis":                         "pericarditis",

    # Respiratory
    "asthma":                               "asthma",
    "chronic obstructive pulmonary disease": "chronic obstructive pulmonary disease",
    "copd":                                 "chronic obstructive pulmonary disease",
    "cystic fibrosis":                      "cystic fibrosis",

    # Rare / orphan
    "achondroplasia":                       "achondroplasia",
    "tuberous sclerosis":                   "tuberous sclerosis",
    "paroxysmal nocturnal hemoglobinuria":  "paroxysmal nocturnal hemoglobinuria",
    "spinal muscular atrophy":              "spinal muscular atrophy",
    "gaucher disease":                      "Gaucher disease",
    "fabry disease":                        "Fabry disease",
    "wilson disease":                       "Wilson disease",

    # GI
    "peptic ulcer":                         "peptic ulcer disease",
    "peptic ulcer disease":                 "peptic ulcer disease",
    "gastroesophageal reflux disease":      "gastroesophageal reflux disease",
    "gerd":                                 "gastroesophageal reflux disease",

    # Renal
    "chronic kidney disease":               "chronic kidney disease",
    "acute kidney injury":                  "acute kidney tubular necrosis",

    # Pain
    "pain":                                 "pain",
    "acute pain":                           "pain",
    "chronic pain":                         "chronic pain disorder",
    "fibromyalgia":                         "fibromyalgia",
}


# ─────────────────────────────────────────────────────────────────────────────
# FIX 6b: Text normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

ABBREVIATION_MAP = {
    r"\bhiv\b":   "human immunodeficiency virus",
    r"\baids\b":  "acquired immunodeficiency syndrome",
    r"\bcopd\b":  "chronic obstructive pulmonary disease",
    r"\badhd\b":  "attention deficit hyperactivity disorder",
    r"\bgerd\b":  "gastroesophageal reflux disease",
    r"\bcml\b":   "chronic myelogenous leukemia",
    r"\bcll\b":   "chronic lymphocytic leukemia",
    r"\ball\b":   "acute lymphoblastic leukemia",
    r"\baml\b":   "acute myeloid leukemia",
    r"\bms\b":    "multiple sclerosis",
    r"\bra\b":    "rheumatoid arthritis",
    r"\bsle\b":   "systemic lupus erythematosus",
    r"\bibd\b":   "inflammatory bowel disease",
    r"\bpah\b":   "pulmonary arterial hypertension",
    r"\bpnh\b":   "paroxysmal nocturnal hemoglobinuria",
    r"\bsma\b":   "spinal muscular atrophy",
    r"\bt2dm\b":  "type 2 diabetes mellitus",
    r"\bt1dm\b":  "type 1 diabetes mellitus",
}


def normalize_repodb_name(name: str) -> str:
    """
    Normalize a RepoDB disease name for OpenTargets lookup.

    Steps:
    1. Lowercase and strip whitespace
    2. Remove possessives ("Alzheimer's" → "Alzheimer")
    3. Expand known abbreviations
    4. Check REPODB_TO_EFO map
    """
    name = name.lower().strip()
    # Remove possessives
    name = re.sub(r"'s\b", "", name)
    name = re.sub(r"'s\b", "", name)  # handle curly apostrophe
    # Expand abbreviations
    for pattern, replacement in ABBREVIATION_MAP.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    # Check hardcoded map
    if name in REPODB_TO_EFO:
        return REPODB_TO_EFO[name]
    return name


# ─────────────────────────────────────────────────────────────────────────────
# FIX 6c: OpenTargets text-search fallback
# ─────────────────────────────────────────────────────────────────────────────

_OT_SEARCH_CACHE: Dict[str, Optional[str]] = {}
_OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_OT_SEARCH_QUERY = """
query SearchDisease($query: String!) {
  search(queryString: $query, entityNames: ["disease"], page: {index: 0, size: 3}) {
    hits {
      id
      name
      score
    }
  }
}
"""


async def opentargets_search(disease_name: str) -> Optional[str]:
    """
    Query OpenTargets text search and return the top disease name if confident.
    Caches results in _OT_SEARCH_CACHE to avoid repeated API calls.
    """
    if disease_name in _OT_SEARCH_CACHE:
        return _OT_SEARCH_CACHE[disease_name]

    try:
        import aiohttp
        payload = json.dumps({
            "query": _OT_SEARCH_QUERY,
            "variables": {"query": disease_name},
        }).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        async with aiohttp.ClientSession() as sess:
            async with sess.post(_OT_GRAPHQL_URL, data=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    _OT_SEARCH_CACHE[disease_name] = None
                    return None
                data = await resp.json()

        hits = data.get("data", {}).get("search", {}).get("hits", [])
        if hits and hits[0].get("score", 0) > 0.5:
            result = hits[0]["name"]
            logger.info(f"  OT search: '{disease_name}' → '{result}' (score={hits[0]['score']:.2f})")
            _OT_SEARCH_CACHE[disease_name] = result
            return result
    except Exception as e:
        logger.debug(f"  OT search failed for '{disease_name}': {e}")

    _OT_SEARCH_CACHE[disease_name] = None
    return None


async def resolve_disease_name(pipeline, raw_name: str) -> Tuple[Optional[Dict], str]:
    """
    Try to resolve a RepoDB disease name to OpenTargets disease data.

    Three-layer strategy:
    1. Hardcoded REPODB_TO_EFO map + text normalization
    2. Original name as-is
    3. OpenTargets text-search fallback

    Returns (disease_data, resolved_name_used).
    """
    # Layer 1: normalize + map
    normalized = normalize_repodb_name(raw_name)
    disease_data = await pipeline.data_fetcher.fetch_disease_data(normalized)
    if disease_data:
        return disease_data, normalized

    # Layer 2: try original name as-is (handles cases already in EFO format)
    if normalized != raw_name.lower().strip():
        disease_data = await pipeline.data_fetcher.fetch_disease_data(raw_name)
        if disease_data:
            return disease_data, raw_name

    # Layer 3: OpenTargets text-search
    ot_name = await opentargets_search(normalized)
    if ot_name and ot_name != normalized:
        disease_data = await pipeline.data_fetcher.fetch_disease_data(ot_name)
        if disease_data:
            return disease_data, ot_name

    return None, normalized


# ─────────────────────────────────────────────────────────────────────────────
# Auto-fetch RepoDB CSV
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# FIX 7: Disease area filter
#
# Skip disease areas where gene/pathway scoring has low signal by design:
# - Infectious/bacterial/viral: no gene overlap with OpenTargets gene sets
# - Acute poisoning / toxicology: not in EFO
# - Surgical conditions: not molecularly tractable
#
# SKIP_DISEASE_KEYWORDS: diseases whose names contain any of these strings
# (case-insensitive) are filtered OUT when --disease-filter is active.
# ─────────────────────────────────────────────────────────────────────────────

SKIP_DISEASE_KEYWORDS = [
    # Infectious
    "infection", "infectious", "bacterial", "viral", "fungal", "parasit",
    "abscess", "sepsis", "pneumonia", "meningitis", "encephalitis",
    "hepatitis", "tuberculosis", "malaria", "hiv", "aids", "candida",
    "aspergill", "actinomycosis", "keratitis", "acinetobacter",
    # Acute / toxicology
    "poisoning", "overdose", "toxicity", "toxic", "accidental",
    # Surgical / trauma
    "fracture", "wound", "burn", "injury", "trauma", "post-operative",
    "surgical", "perioperative",
    # Purely symptomatic
    "nausea", "vomiting", "constipation", "diarrhea", "diarrhoea",
]

# Disease areas to INCLUDE when --disease-filter is used.
# A disease passes if its name contains at least one include keyword
# AND none of the skip keywords above.
INCLUDE_DISEASE_KEYWORDS = {
    "oncology":       ["cancer", "carcinoma", "leukemia", "lymphoma", "myeloma",
                       "tumor", "tumour", "sarcoma", "glioma", "melanoma",
                       "neoplasm", "malignant", "adenocarcinoma"],
    "cardiovascular": ["cardiac", "heart", "coronary", "myocardial", "atrial",
                       "hypertension", "arterial", "vascular", "stroke",
                       "thrombosis", "angina", "arrhythmia", "fibrillation",
                       "atherosclerosis", "hyperlipidemia", "hypercholesterol"],
    "metabolic":      ["diabetes", "obesity", "metabolic", "thyroid", "gout",
                       "osteoporosis", "hypercholesterol", "dyslipidemia",
                       "cushing", "polycystic", "adrenal"],
    "neurological":   ["alzheimer", "parkinson", "epilepsy", "sclerosis",
                       "neuropath", "migraine", "schizophrenia", "bipolar",
                       "depression", "anxiety", "dementia", "huntington",
                       "adhd", "autism", "ataxia"],
    "autoimmune":     ["arthritis", "lupus", "inflammatory", "psoriasis",
                       "spondylitis", "crohn", "colitis", "autoimmune",
                       "vasculitis", "sjogren", "scleroderma", "myositis"],
    "respiratory":    ["asthma", "pulmonary", "copd", "fibrosis", "bronchitis",
                       "emphysema", "respiratory"],
    "rare":           ["syndrome", "disease", "disorder", "dystrophy",
                       "deficiency", "atrophy", "dysplasia"],
}


def passes_disease_filter(disease_name: str, filter_areas: Optional[List[str]]) -> bool:
    """
    Return True if disease_name should be included in the benchmark.

    If filter_areas is None, all diseases pass.
    Otherwise:
      - Skip if name contains any SKIP_DISEASE_KEYWORDS
      - Keep if name contains any keyword for the requested areas
    """
    if filter_areas is None:
        return True

    name_lower = disease_name.lower()

    # Always skip low-signal disease types
    for kw in SKIP_DISEASE_KEYWORDS:
        if kw in name_lower:
            return False

    # Keep if it matches any requested area
    for area in filter_areas:
        area_kws = INCLUDE_DISEASE_KEYWORDS.get(area.lower(), [area.lower()])
        for kw in area_kws:
            if kw in name_lower:
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# FIX 8: ChEMBL synonym lookup cache
#
# Query ChEMBL molecule synonyms API for a drug name and return all synonyms.
# Used as layer 4 in match_drug_name when salt-stripping and substring matching
# both fail. Results cached to avoid redundant API calls.
# ─────────────────────────────────────────────────────────────────────────────

_CHEMBL_SYNONYM_CACHE: Dict[str, List[str]] = {}
_CHEMBL_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={}&limit=3"


def _fetch_chembl_synonyms(drug_name: str) -> List[str]:
    """
    Fetch ChEMBL synonyms for a drug name. Returns list of normalized synonym strings.
    Uses stdlib urllib — no extra deps.
    """
    if drug_name in _CHEMBL_SYNONYM_CACHE:
        return _CHEMBL_SYNONYM_CACHE[drug_name]

    synonyms: List[str] = []
    try:
        url = _CHEMBL_SEARCH_URL.format(urllib.request.quote(drug_name))
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        for mol in data.get("molecules", []):
            # Preferred name
            pref = mol.get("pref_name") or ""
            if pref:
                synonyms.append(pref.lower().strip())
            # All synonyms
            for syn in mol.get("molecule_synonyms", []):
                s = syn.get("molecule_synonym") or syn.get("synonyms") or ""
                if s:
                    synonyms.append(s.lower().strip())
    except Exception:
        pass  # Network failure → fall through to no synonyms

    _CHEMBL_SYNONYM_CACHE[drug_name] = synonyms
    return synonyms


def fetch_repodb_if_missing(path: str) -> None:
    if Path(path).exists():
        logger.info(f"RepoDB file found at {path}, skipping download.")
        return

    for url in (_REPODB_PRIMARY_URL, _REPODB_FALLBACK_URL):
        try:
            logger.info(f"Downloading RepoDB CSV from {url} ...")
            urllib.request.urlretrieve(url, path)
            logger.info(f"RepoDB CSV saved to {path}")
            return
        except Exception as exc:
            logger.warning(f"  Failed ({exc}), trying next URL...")

    Path(path).unlink(missing_ok=True)
    logger.error(
        "Could not download RepoDB from any known URL.\n"
        "Please download manually from https://unmtid-shinyapps.net/shiny/repodb/ "
        f"and place the file at: {path}"
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# RepoDB data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_repodb(path: str) -> Tuple[List[Dict], List[str]]:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = 1 if row.get("status", "").lower() == "approved" else 0
            pairs.append({
                "drug":    row["drug_name"].strip(),
                "disease": row["ind_name"].strip(),
                "label":   label,
            })

    diseases = sorted(set(p["disease"] for p in pairs))
    logger.info(f"Loaded {len(pairs)} pairs across {len(diseases)} diseases from {path}")
    return pairs, diseases


def normalize_drug_name(name: str) -> str:
    """
    Normalize a drug name for fuzzy matching.
    Strips salt suffixes, punctuation variants, and formulation descriptors
    that differ between RepoDB (clinical names) and ChEMBL (INN names).
    """
    name = name.lower().strip()
    # Remove common salt / formulation suffixes
    for suffix in (
        " hydrochloride", " hcl", " sodium", " potassium", " calcium",
        " mesylate", " maleate", " tartrate", " acetate", " sulfate",
        " sulphate", " phosphate", " fumarate", " succinate", " citrate",
        " bromide", " chloride", " iodide", " nitrate", " gluconate",
        " anhydrous", " monohydrate", " dihydrate", " trihydrate",
        " hemihydrate", " sesquihydrate", " base", " free acid", " free base",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    # Normalize punctuation
    name = re.sub(r"[-\u2013]", " ", name)  # hyphens/en-dashes → space
    name = re.sub(r"\s+", " ", name)        # collapse whitespace
    return name.strip()


def match_drug_name(candidate_name: str, repodb_name: str) -> bool:
    """
    Match pipeline candidate name against RepoDB drug name.

    Checks (in order):
    1. Exact match after normalization
    2. Substring match (handles "imatinib mesylate" vs "imatinib")
    3. First-token match for multi-word names (min 5 chars to avoid false positives)
    4. ChEMBL synonym lookup — catches brand/INN name mismatches
       e.g. "Gleevec" (RepoDB) vs "imatinib" (ChEMBL)
    """
    a = normalize_drug_name(candidate_name)
    b = normalize_drug_name(repodb_name)

    # Layer 1: exact
    if a == b:
        return True
    # Layer 2: substring
    if a in b or b in a:
        return True
    # Layer 3: first token (min 5 chars)
    a_first = a.split()[0] if a.split() else a
    b_first = b.split()[0] if b.split() else b
    if len(a_first) >= 5 and a_first == b_first:
        return True
    # Layer 4: ChEMBL synonyms for candidate_name
    syns = _fetch_chembl_synonyms(a)
    b_norm = normalize_drug_name(repodb_name)
    for syn in syns:
        syn_norm = normalize_drug_name(syn)
        if syn_norm == b_norm or syn_norm in b_norm or b_norm in syn_norm:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# AUC helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auc_roc(scores: List[float], labels: List[int]) -> float:
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = fp = auc = prev_tp = prev_fp = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        auc    += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_tp = tp
        prev_fp = fp

    return auc / (n_pos * n_neg)


def _auc_pr(scores: List[float], labels: List[int]) -> float:
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")

    tp = fp = auc = 0.0
    prev_recall = 0.0
    prev_prec   = 1.0

    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        prec   = tp / (tp + fp)
        auc   += (recall - prev_recall) * (prec + prev_prec) / 2.0
        prev_recall = recall
        prev_prec   = prec

    return auc


def _hit_at_n(rank: Optional[int], n: int) -> int:
    return 1 if rank is not None and rank <= n else 0


def _reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_benchmark(
    repodb_path:    str,
    top_n:          int            = 50,
    output_path:    str            = "repodb_benchmark_results.json",
    max_diseases:   Optional[int]  = None,
    disease_filter: Optional[List[str]] = None,   # FIX 7
    fetch_pubmed:   bool           = False,        # FIX 9
) -> Dict:
    fetch_repodb_if_missing(repodb_path)
    pairs, diseases = load_repodb(repodb_path)

    if max_diseases:
        diseases = diseases[:max_diseases]
        logger.info(f"Limiting to first {max_diseases} diseases")

    # FIX 7: Apply disease area filter
    if disease_filter:
        before = len(diseases)
        diseases = [d for d in diseases if passes_disease_filter(d, disease_filter)]
        logger.info(
            f"Disease filter '{','.join(disease_filter)}': "
            f"{len(diseases)}/{before} diseases kept"
        )

    pipeline = ProductionPipeline()

    n_resolved   = 0
    n_skipped    = 0
    skip_reasons = []

    try:
        logger.info("Fetching approved drugs from ChEMBL (shared across all diseases)...")
        drugs_data = await pipeline.fetch_approved_drugs(limit=3000)
        logger.info(f"Using {len(drugs_data)} drugs\n")

        all_scores: List[float] = []
        all_labels: List[int]   = []
        per_disease_results      = []
        hit_at_n_counts          = []
        reciprocal_ranks         = []

        for d_idx, disease_name in enumerate(diseases, 1):
            logger.info(f"[{d_idx}/{len(diseases)}] Benchmarking: {disease_name}")

            # FIX 6: Use three-layer resolution instead of raw name lookup
            disease_data, resolved_name = await resolve_disease_name(pipeline, disease_name)

            if not disease_data:
                logger.warning(f"  Skipped — could not resolve '{disease_name}' in OpenTargets")
                n_skipped += 1
                skip_reasons.append(disease_name)
                continue

            n_resolved += 1
            if resolved_name.lower() != disease_name.lower():
                logger.info(f"  Resolved: '{disease_name}' → '{resolved_name}'")

            candidates = await pipeline.generate_candidates(
                disease_data=disease_data,
                drugs_data=drugs_data,
                min_score=0.0,
                fetch_pubmed=fetch_pubmed,   # FIX 9: controllable via CLI flag
            )

            candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)

            disease_pairs  = [p for p in pairs if p["disease"] == disease_name]
            positive_drugs = {
                normalize_drug_name(p["drug"])
                for p in disease_pairs
                if p["label"] == 1
            }

            disease_scores: List[float] = []
            disease_labels: List[int]   = []

            for rank, cand in enumerate(candidates_sorted, 1):
                matched_positive = any(
                    match_drug_name(cand["name"], pos_drug)
                    for pos_drug in positive_drugs
                )
                label = 1 if matched_positive else 0
                disease_scores.append(cand["score"])
                disease_labels.append(label)

            all_scores.extend(disease_scores)
            all_labels.extend(disease_labels)

            for pos_drug in positive_drugs:
                matched_rank = None
                for rank, cand in enumerate(candidates_sorted, 1):
                    if match_drug_name(cand["name"], pos_drug):
                        matched_rank = rank
                        break
                hit_at_n_counts.append(_hit_at_n(matched_rank, top_n))
                reciprocal_ranks.append(_reciprocal_rank(matched_rank))

            disease_auc = _auc_roc(disease_scores, disease_labels)
            disease_pr  = _auc_pr(disease_scores, disease_labels)

            per_disease_results.append({
                "disease":         disease_name,
                "resolved_as":     resolved_name,
                "n_positives":     len(positive_drugs),
                "n_candidates":    len(candidates),
                "auc_roc":         round(disease_auc, 4) if disease_auc == disease_auc else None,
                "auc_pr":          round(disease_pr,  4) if disease_pr  == disease_pr  else None,
            })

            import math
            auc_roc_str = f"{disease_auc:.3f}" if not math.isnan(disease_auc) else "N/A"
            auc_pr_str  = f"{disease_pr:.3f}"  if not math.isnan(disease_pr)  else "N/A"
            logger.info(
                f"  Positives: {len(positive_drugs)}, "
                f"AUC-ROC: {auc_roc_str}, "
                f"AUC-PR: {auc_pr_str}"
            )

    finally:
        await pipeline.close()

    global_auc_roc = _auc_roc(all_scores, all_labels)
    global_auc_pr  = _auc_pr(all_scores, all_labels)
    hit_at_n       = sum(hit_at_n_counts) / len(hit_at_n_counts) if hit_at_n_counts else 0.0
    mrr            = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    logger.info("\n" + "=" * 70)
    logger.info("REPODB BENCHMARK RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Diseases requested: {len(diseases)}")
    logger.info(f"  Diseases resolved:  {n_resolved}  ({n_resolved/len(diseases):.0%})")
    logger.info(f"  Diseases skipped:   {n_skipped}  (name not in OpenTargets)")
    logger.info(f"  Global AUC-ROC:     {global_auc_roc:.4f}")
    logger.info(f"  Global AUC-PR:      {global_auc_pr:.4f}")
    logger.info(f"  Hit@{top_n}:            {hit_at_n:.4f}")
    logger.info(f"  MRR:                {mrr:.4f}")

    if skip_reasons:
        logger.info(f"\n  Skipped diseases (add to REPODB_TO_EFO map to fix):")
        for name in skip_reasons:
            logger.info(f"    - {name}")

    output = {
        "summary": {
            "n_diseases_requested": len(diseases),
            "n_diseases_resolved":  n_resolved,
            "n_diseases_skipped":   n_skipped,
            "resolve_rate":         round(n_resolved / len(diseases), 4) if diseases else 0,
            "disease_filter":       disease_filter,
            "fetch_pubmed":         fetch_pubmed,
            "n_drugs":              len(drugs_data),
            "top_n":                top_n,
            "global_auc_roc":       round(global_auc_roc, 4),
            "global_auc_pr":        round(global_auc_pr,  4),
            f"hit_at_{top_n}":      round(hit_at_n, 4),
            "mrr":                  round(mrr, 4),
        },
        "skipped_diseases":    skip_reasons,
        "per_disease_results": per_disease_results,
    }

    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults written to: {out_path.resolve()}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark drug repurposing pipeline against RepoDB"
    )
    parser.add_argument("--repodb-file",      type=str,  default="repodb.csv")
    parser.add_argument("--top-n",            type=int,  default=50)
    parser.add_argument("--output",           type=str,  default="repodb_benchmark_results.json")
    parser.add_argument("--max-diseases",     type=int,  default=None)
    parser.add_argument(
        "--disease-filter",
        type=str, default=None,
        help=(
            "Comma-separated disease areas to include. "
            "Options: oncology, cardiovascular, metabolic, neurological, autoimmune, respiratory, rare. "
            "Example: --disease-filter oncology,cardiovascular,metabolic"
        ),
    )
    parser.add_argument(
        "--fetch-pubmed",
        action="store_true", default=False,
        help="Enable PubMed literature scoring (slower but improves Hit@N/MRR).",
    )
    args = parser.parse_args()

    disease_filter = (
        [a.strip() for a in args.disease_filter.split(",")]
        if args.disease_filter else None
    )

    try:
        asyncio.run(run_benchmark(
            repodb_path=args.repodb_file,
            top_n=args.top_n,
            output_path=args.output,
            max_diseases=args.max_diseases,
            disease_filter=disease_filter,
            fetch_pubmed=args.fetch_pubmed,
        ))
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()