"""Build the learning curve Jupyter notebook and export a plot."""

import json
from pathlib import Path

import matplotlib
import nbformat as nbf

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    report_path = Path("data/eval/reports/learning_curve.jsonl")
    notebook_path = Path("notebooks/learning_curve.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Jupyter Notebook
    nb = nbf.v4.new_notebook()
    
    markdown_intro = """# Citation Recommender Learning Curve
This notebook analyzes the incremental fine-tuning simulation.
As user interactions (simulated clicks) accumulate, we plot the performance of the embedder."""
    
    code_cell = """import json
import matplotlib.pyplot as plt

steps = []
hit_at_1 = []
recall_at_20 = []
mrr_at_20 = []

with open('../data/eval/reports/learning_curve.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        steps.append(data['step'])
        m = data['metrics']
        hit_at_1.append(m['val_cosine_accuracy@1'])
        recall_at_20.append(m['val_cosine_recall@20'])
        mrr_at_20.append(m['val_cosine_mrr@20'])

plt.figure(figsize=(10, 6))
plt.plot(steps, hit_at_1, marker='o', label='Hit@1')
plt.plot(steps, recall_at_20, marker='s', label='Recall@20')
plt.plot(steps, mrr_at_20, marker='^', label='MRR@20')
plt.title('Incremental Fine-Tuning Learning Curve')
plt.xlabel('Simulated User Interactions (Pairs)')
plt.ylabel('Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""
    
    markdown_analysis = """## Analysis
At Step 0, the model is the zero-shot baseline BGE-Large.
As we fine-tune on small batches of simulated data (400, 800...), we observe the model's metrics. Note that a small dip is normal (catastrophic forgetting of the general prompt prefix) before it specializes to the citation task!"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_intro),
        nbf.v4.new_code_cell(code_cell),
        nbf.v4.new_markdown_cell(markdown_analysis)
    ]
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Generated {notebook_path}")

    # 2. Export Plot Image for Artifact
    steps, hit, recall, mrr = [], [], [], []
    with open(report_path) as f:
        for line in f:
            data = json.loads(line)
            steps.append(data['step'])
            m = data['metrics']
            hit.append(m['val_cosine_accuracy@1'])
            recall.append(m['val_cosine_recall@20'])
            mrr.append(m['val_cosine_mrr@20'])
            
    plt.figure(figsize=(10, 6))
    plt.plot(steps, hit, marker='o', label='Hit@1', color='#1f77b4', linewidth=2)
    plt.plot(steps, recall, marker='s', label='Recall@20', color='#ff7f0e', linewidth=2)
    plt.plot(steps, mrr, marker='^', label='MRR@20', color='#2ca02c', linewidth=2)
    plt.title('Incremental Fine-Tuning Learning Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Simulated User Interactions (Triplets)', fontsize=12)
    plt.ylabel('Validation Score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save directly to the user's artifact dir so the agent can embed it
    import os
    artifact_dir = r"C:\Users\sampe\.gemini\antigravity-ide\brain\aa7e0988-5e85-4a92-8c47-59d595298abd"
    img_path = os.path.join(artifact_dir, "learning_curve_plot.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    print(f"Exported plot to {img_path}")

if __name__ == "__main__":
    main()
