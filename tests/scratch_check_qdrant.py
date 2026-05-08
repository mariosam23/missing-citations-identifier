"""Quick check: Qdrant payload schema and abstract availability."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient

c = QdrantClient("http://localhost:6333")
pts, _ = c.scroll("papers", limit=5, with_payload=True)

print(f"=== Sampled {len(pts)} points from 'papers' collection ===\n")
for p in pts:
    payload = p.payload or {}
    keys = sorted(payload.keys())
    abstract = payload.get("abstract")
    has_abstract = abstract is not None and len(str(abstract).strip()) > 0
    print(f"  ID:           {p.id}")
    print(f"  Payload keys: {keys}")
    print(f"  Title:        {(payload.get('title') or '')[:80]}")
    print(f"  Has abstract: {has_abstract}")
    if has_abstract:
        print(f"  Abstract:     {str(abstract)[:100]}...")
    print()
