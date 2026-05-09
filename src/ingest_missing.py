import requests
import datetime

from utils.logger import logger
from database.postgres.engine import get_session
from database.postgres.tables.paper import Paper
from indexer import EmbeddingIndex


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract string from OpenAlex's inverted index."""
    if not inverted_index:
        return ""
    try:
        max_idx = max(max(positions) for positions in inverted_index.values())
        words = [""] * (max_idx + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words).strip()
    except Exception as e:
        logger.warning(f"Failed to rebuild abstract: {e}")
        return ""


def ingest_openalex_papers(openalex_ids: list[str], embedding_index: EmbeddingIndex) -> int:
    """Fetch missing papers from OpenAlex, save to Postgres, and index in Qdrant."""
    if not openalex_ids:
        return 0

    # Clean IDs to just the 'W...' format
    clean_ids = []
    for aid in openalex_ids:
        if not aid:
            continue
        clean_id = str(aid).replace("https://openalex.org/", "").strip().upper()
        clean_ids.append(clean_id)
        
    if not clean_ids:
        return 0
    
    # OpenAlex allows querying multiple IDs using filter=openalex:W123|W456
    batch_size = 50
    papers_to_insert = []
    papers_for_qdrant = []
    
    for i in range(0, len(clean_ids), batch_size):
        batch_ids = clean_ids[i:i+batch_size]
        id_filter = "|".join(batch_ids)
        url = f"https://api.openalex.org/works?filter=openalex:{id_filter}&per-page={len(batch_ids)}"
        
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                logger.error(f"OpenAlex request failed with status {resp.status_code}")
                continue
                
            data = resp.json()
            for work in data.get("results", []):
                paper_id = work.get("id", "").replace("https://openalex.org/", "").upper()
                if not paper_id:
                    continue
                    
                doi_url = work.get("doi")
                doi = doi_url.replace("https://doi.org/", "").lower() if doi_url else None
                title = work.get("title", "")
                
                abstract_index = work.get("abstract_inverted_index")
                abstract = rebuild_abstract(abstract_index)
                
                # Parse publication_date to a Python date object if possible
                pub_date_str = work.get("publication_date")
                pub_date = None
                if pub_date_str:
                    try:
                        pub_date = datetime.datetime.strptime(pub_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        pass
                
                cited_by = work.get("cited_by_count", 0)
                ref_count = work.get("referenced_works_count", 0)
                
                # Try to extract venue from primary_location
                venue = None
                primary_loc = work.get("primary_location")
                if primary_loc and primary_loc.get("source"):
                    venue = primary_loc["source"].get("display_name")
                
                # Create ORM object for Postgres
                paper = Paper(
                    paperId=paper_id,
                    doi=doi,
                    title=title,
                    abstract=abstract,
                    publication_date=pub_date,
                    cited_by_count=cited_by,
                    referenced_works_count=ref_count,
                    paper_type=work.get("type"),
                    venue=venue,
                )
                papers_to_insert.append(paper)
                
                # Create dict for Qdrant
                papers_for_qdrant.append({
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "year": pub_date.year if pub_date else work.get("publication_year"),
                    "venue": venue,
                    "cited_by_count": cited_by,
                })
                
        except Exception as e:
            logger.error(f"Failed to fetch batch from OpenAlex: {e}")
            
    if not papers_to_insert:
        logger.info("No valid papers fetched from OpenAlex.")
        return 0
        
    # 1. Insert to PostgreSQL
    inserted_count = 0
    with get_session() as session:
        for paper in papers_to_insert:
            existing = session.get(Paper, paper.paperId)
            if not existing:
                session.add(paper)
                inserted_count += 1
        session.commit()
    logger.info(f"Inserted {inserted_count} new papers into PostgreSQL.")
    
    # 2. Insert to Qdrant (EmbeddingIndex handles skips automatically)
    indexed_count = embedding_index.upsert_papers(papers_for_qdrant, skip_existing=True)
    logger.info(f"Embedded and indexed {indexed_count} papers into Qdrant.")
    
    return inserted_count
