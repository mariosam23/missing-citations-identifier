import sys
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# Add src to Python path
sys.path.append(str(Path.cwd().parent / "src"))

from pipeline.citation_graph import build_citation_graph_from_strings


# Sample references (e.g., extracted from a PDF's bibliography)
sample_references = [
    "Attention Is All You Need, Vaswani et al., 2017",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., 2018",
    "Language Models are Few-Shot Learners, Brown et al., 2020",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach, Liu et al., 2019",
    "Transformers: State-of-the-Art Natural Language Processing, Wolf et al., 2020"
]

# Build the citation graph
print("Building citation graph...")
print("This might take a few seconds due to OpenAlex rate limits.\\n")

cg = build_citation_graph_from_strings(sample_references)

print(f"Graph built!")
print(f"Nodes (Papers): {cg.graph.number_of_nodes()}")
print(f"Edges (Couplings): {cg.graph.number_of_edges()}")

# Visualize the Citation Graph
plt.figure(figsize=(14, 9))

G = cg.graph
pos = nx.spring_layout(G, k=2.0, seed=42)

# Use paper titles for node labels (truncate if too long)
labels = {node: data.get('title', node)[:30] + "..." for node, data in G.nodes(data=True)}

# Edge weights based on number of shared references
edges = G.edges()
weights = [G[u][v].get('weight', 1) for u, v in edges]

nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#ffcc99', edgecolors='#555555')
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.5, edge_color='#888888')

# Draw labels slightly above nodes
pos_labels = {node: (coords[0], coords[1] + 0.08) for node, coords in pos.items()}
bbox_props = dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#cccccc", lw=1.5, alpha=0.9)
nx.draw_networkx_labels(G, pos_labels, labels=labels, font_size=10, font_weight='bold', bbox=bbox_props)

plt.title("Citation Graph (Bibliographic Coupling)", fontsize=18, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.margins(0.2)
plt.show()


# Inspect a specific edge to see their shared references
edges_data = list(G.edges(data=True))
if edges_data:
    u, v, data = edges_data[0]
    title_u = G.nodes[u].get('title', u)
    title_v = G.nodes[v].get('title', v)
    
    print(f"Edge between:\n- {title_u}\n- {title_v}\n")
    print(f"Coupling Weight (Shared References): {data.get('weight', 1)}")
    print(f"Shared Reference IDs (S2 Paper IDs): {data.get('shared_refs', [])}")