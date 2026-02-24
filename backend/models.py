from pydantic import BaseModel
from typing import Optional


class DrugCandidate(BaseModel):
    drug_name: str
    drug_id: str
    original_indication: str
    composite_score: float
    pathway_overlap_score: float
    gene_target_score: float
    literature_score: float
    shared_genes: list[str]
    shared_pathways: list[str]
    mechanism: str
    explanation: str
    confidence: str


class QueryRequest(BaseModel):
    disease_name: str
    top_k: int = 10
    min_score: float = 0.3
    anthropic_api_key: Optional[str] = None


class RepurposingResult(BaseModel):
    disease_name: str
    disease_genes: list[str]
    disease_pathways: list[str]
    candidates: list[DrugCandidate]
    graph_stats: dict
    data_sources: list[str]