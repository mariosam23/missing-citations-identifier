import networkx as nx

class ConceptGraphBuilder:
    def __int__(self):
        pass
        

    def build_graph(self, concepts: list[dict]) -> nx.Graph:
        G = nx.Graph()

        for concept in concepts:
            G.add_node(concept['label'], score=concept['score'])
        
        # Add simple fully-connected edges weighted by combined salience
        # In the MVP, just assume if they are in the same abstract, they interact
        labels = [c['label'] for c in concepts]

        for i, l1 in enumerate(labels):
            for j, l2 in enumerate(labels):
                if i < j:
                    w1 = G.nodes[l1]['score']
                    w2 = G.nodes[l2]['score']
                    G.add_edge(l1, l2, weight=(w1 * w2))
        
        return G

    def get_central_concepts(self, G: nx.Graph, top_n: int = 5) -> list[str]:
        if len(G) == 0:
            return []
        
        pagerank = nx.pagerank(G, weight='weight')
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:top_n]]