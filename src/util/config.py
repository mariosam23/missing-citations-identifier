import os

class FetchDataConfig:
    SEARCH_QUERY = 'cat:cs.AI OR cat:cs.CL OR cat:cs.FL'
    MAX_RESULTS = 1000
    DATA_DIR = "../../data"
    OUTPUT_FILE = os.path.join(DATA_DIR, "arxiv_papers.json")
