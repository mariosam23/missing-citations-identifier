import arxiv
import json
import os

from config import FetchDataConfig


def fetch_arxiv_papers():
    client = arxiv.Client()

    if not os.path.exists(FetchDataConfig.DATA_DIR):
        os.makedirs(FetchDataConfig.DATA_DIR)

    print(f"Fetching {FetchDataConfig.MAX_RESULTS} papers from ArXiv...")
    
    search = arxiv.Search(
        query=FetchDataConfig.SEARCH_QUERY,
        max_results=FetchDataConfig.MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers_data = []

    # Iterate through results
    for result in client.results(search):
        # We only save what we need to keep the file small
        paper_info = {
            "id": result.entry_id,          # The ArXiv URL
            "title": result.title,
            "abstract": result.summary.replace("\n", " "), # Clean up newlines
            "published": str(result.published),
            "authors": [a.name for a in result.authors],
            "citation_count": 0
        }
        papers_data.append(paper_info)
        
        if len(papers_data) % 100 == 0:
            print(f"Downloaded {len(papers_data)}...")

    with open(FetchDataConfig.OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(papers_data, f, indent=2)

    print(f"Saved {len(papers_data)} papers to {FetchDataConfig.OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_arxiv_papers()
