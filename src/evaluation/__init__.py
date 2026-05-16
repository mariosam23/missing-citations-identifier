"""Phase 7 — Offline citation recovery benchmark.

Provides a reproducible evaluation harness: frozen train/val/test splits of
citing papers, four standard IR metrics (Hit@K, Recall@K, MRR@K, NDCG@K),
a ``Variant`` protocol for pluggable retrieval strategies, and CLI tooling
for running evaluations and comparing reports.
"""
