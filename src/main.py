from database import init_db
from pipeline.pdf_parser import NougatPDFParser
import json

def main():
    # init_db()
    # fetch_and_store_papers()
    parser = NougatPDFParser("../papers/BERT.pdf")
    paper = parser.parse()

    with open("raw_nougat_output.txt", "w", encoding="utf-8") as f:
        f.write(parser._last_markdown)

    with open("paper.json", "w") as f:
        json.dump(paper.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
