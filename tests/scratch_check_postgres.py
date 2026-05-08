"""Check how many Postgres papers have abstracts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import func, select
from database.postgres.engine import get_session
from database.postgres.tables import Paper

with get_session() as session:
    total = session.scalar(select(func.count(Paper.paperId)))
    with_abstract = session.scalar(
        select(func.count(Paper.paperId)).where(
            Paper.abstract.is_not(None),
            Paper.abstract != "",
        )
    )
    print(f"Total papers in Postgres: {total}")
    print(f"Papers with abstract:     {with_abstract}")
    print(f"Papers without abstract:  {total - with_abstract}")
    
    # sample a few
    samples = session.execute(
        select(Paper.paperId, Paper.title, Paper.abstract)
        .where(Paper.abstract.is_not(None), Paper.abstract != "")
        .limit(3)
    ).all()
    print("\n=== Sample papers with abstracts ===")
    for pid, title, abstract in samples:
        print(f"  ID: {pid}")
        print(f"  Title: {(title or '')[:80]}")
        print(f"  Abstract: {(abstract or '')[:120]}...")
        print()
