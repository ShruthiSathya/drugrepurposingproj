"""
PRODUCTION GRAPH BUILDER
Builds knowledge graphs from real API data (OpenTargets, ChEMBL, DGIdb)
"""

import networkx as nx
from typing import Dict, List, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionGraphBuilder:
    """
    Builds heterogeneous knowledge graph from disease and drug data.
    Uses real data from public databases.
    """
    
    def __init__(self):
        self.graph = None
    
    def build_graph(self, disease_data: Dict, drugs_data: List[Dict]) -> nx.Graph:
        """
        Build multi-partite graph: Disease -> Genes <- Drugs
                                    Disease -> Pathways <- Drugs
        
        Args:
            disease_data: Disease info from OpenTargets
            drugs_data: Drug info from ChEMBL/DGIdb
            
        Returns:
            NetworkX MultiGraph with disease-gene-drug-pathway relationships
        """
        logger.info("🔨 Building knowledge graph from API data...")
        
        # Create empty graph
        G = nx.Graph()
        
        # Add disease node
        disease_name = disease_data['name']
        G.add_node(
            disease_name,
            type='disease',
            id=disease_data.get('id', ''),
            description=disease_data.get('description', '')[:200],
            is_rare=disease_data.get('is_rare', False),
            active_trials=disease_data.get('active_trials_count', 0)
        )
        
        # Add gene nodes and disease-gene edges
        disease_genes = disease_data.get('genes', [])
        gene_scores = disease_data.get('gene_scores', {})
        
        for gene in disease_genes:
            # Add gene node if not exists
            if gene not in G:
                G.add_node(gene, type='gene')
            
            # Add disease-gene edge with association score
            score = gene_scores.get(gene, 0.5)
            G.add_edge(
                disease_name,
                gene,
                relation='associated_with',
                score=score,
                source='OpenTargets'
            )
        
        logger.info(f"  Added {len(disease_genes)} disease-associated genes")
        
        # Add pathway nodes and disease-pathway edges
        disease_pathways = disease_data.get('pathways', [])
        for pathway in disease_pathways:
            # Add pathway node if not exists
            if pathway not in G:
                G.add_node(pathway, type='pathway')
            
            # Add disease-pathway edge
            G.add_edge(
                disease_name,
                pathway,
                relation='involves_pathway',
                source='curated'
            )
        
        logger.info(f"  Added {len(disease_pathways)} disease-associated pathways")
        
        # Add drug nodes and drug-gene-pathway edges
        drugs_with_targets = 0
        drugs_with_pathways = 0
        
        for drug in drugs_data:
            drug_name = drug['name']
            
            # Add drug node
            G.add_node(
                drug_name,
                type='drug',
                id=drug.get('id', ''),
                indication=drug.get('indication', ''),
                mechanism=drug.get('mechanism', ''),
                approved=drug.get('approved', False),
                smiles=drug.get('smiles', '')
            )
            
            # Add drug-gene edges
            drug_targets = drug.get('targets', [])
            if drug_targets:
                drugs_with_targets += 1
                
                for target in drug_targets:
                    # Add gene node if not exists
                    if target not in G:
                        G.add_node(target, type='gene')
                    
                    # Add drug-gene edge
                    G.add_edge(
                        drug_name,
                        target,
                        relation='targets',
                        source='DGIdb'
                    )
            
            # Add drug-pathway edges
            drug_pathways = drug.get('pathways', [])
            if drug_pathways:
                drugs_with_pathways += 1
                
                for pathway in drug_pathways:
                    # Add pathway node if not exists
                    if pathway not in G:
                        G.add_node(pathway, type='pathway')
                    
                    # Add drug-pathway edge
                    G.add_edge(
                        drug_name,
                        pathway,
                        relation='modulates_pathway',
                        source='inferred_from_targets'
                    )
        
        logger.info(f"  Added {len(drugs_data)} drugs")
        logger.info(f"    {drugs_with_targets} drugs with gene targets")
        logger.info(f"    {drugs_with_pathways} drugs with pathway annotations")
        
        # Graph statistics
        nodes_by_type = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1
        
        logger.info("📊 Graph statistics:")
        logger.info(f"  Total nodes: {G.number_of_nodes()}")
        logger.info(f"  Total edges: {G.number_of_edges()}")
        for node_type, count in nodes_by_type.items():
            logger.info(f"    {node_type}: {count}")
        
        self.graph = G
        return G
    
    def get_drug_disease_paths(self, drug_name: str, disease_name: str) -> List[List[str]]:
        """
        Find all paths connecting a drug to a disease through genes/pathways.
        """
        if self.graph is None:
            return []
        
        if drug_name not in self.graph or disease_name not in self.graph:
            return []
        
        try:
            # Find all simple paths (up to length 3: drug -> gene/pathway -> disease)
            paths = list(nx.all_simple_paths(
                self.graph,
                source=drug_name,
                target=disease_name,
                cutoff=3
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_shared_genes(self, drug_name: str, disease_name: str) -> Set[str]:
        """Get genes that are both disease-associated and drug-targeted."""
        if self.graph is None:
            return set()
        
        if drug_name not in self.graph or disease_name not in self.graph:
            return set()
        
        # Get drug's target genes
        drug_genes = set()
        for neighbor in self.graph.neighbors(drug_name):
            if self.graph.nodes[neighbor].get('type') == 'gene':
                drug_genes.add(neighbor)
        
        # Get disease's associated genes
        disease_genes = set()
        for neighbor in self.graph.neighbors(disease_name):
            if self.graph.nodes[neighbor].get('type') == 'gene':
                disease_genes.add(neighbor)
        
        # Return intersection
        return drug_genes & disease_genes
    
    def get_shared_pathways(self, drug_name: str, disease_name: str) -> Set[str]:
        """Get pathways that are both disease-relevant and drug-modulated."""
        if self.graph is None:
            return set()
        
        if drug_name not in self.graph or disease_name not in self.graph:
            return set()
        
        # Get drug's pathways
        drug_pathways = set()
        for neighbor in self.graph.neighbors(drug_name):
            if self.graph.nodes[neighbor].get('type') == 'pathway':
                drug_pathways.add(neighbor)
        
        # Get disease's pathways
        disease_pathways = set()
        for neighbor in self.graph.neighbors(disease_name):
            if self.graph.nodes[neighbor].get('type') == 'pathway':
                disease_pathways.add(neighbor)
        
        # Return intersection
        return drug_pathways & disease_pathways
    
    def get_graph_stats(self) -> Dict:
        """Get graph statistics for reporting."""
        if self.graph is None:
            return {}
        
        nodes_by_type = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'nodes_by_type': nodes_by_type,
            'density': nx.density(self.graph),
        }


# Maintain backward compatibility
GraphBuilder = ProductionGraphBuilder