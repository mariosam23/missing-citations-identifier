import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no Tk/display required
import matplotlib.pyplot as plt
import pandas as pd

# Add the src folder to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import GrobidPDFParser
from pipeline import extract_sentences

def run_stage1_experiment():
    project_root = Path(__file__).parent.parent.parent
    papers_dir = project_root / "papers"
    assets_dir = project_root / "thesis" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse two PDFs
    pdf1 = papers_dir / "BERT.pdf"
    pdf2 = papers_dir / "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    
    pdfs_to_test = [pdf1, pdf2]
    
    parsed_papers = {}
    extracted_sentences_dict = {}

    print("--- 1. Parsing PDFs ---")
    for pdf_path in pdfs_to_test:
        if not pdf_path.exists():
            print(f"Warning: File {pdf_path} not found.")
            continue
            
        parser = GrobidPDFParser(str(pdf_path))
        try:
            parsed_paper = parser.parse()
            parsed_papers[pdf_path.name] = parsed_paper
            print(f"Parsed {pdf_path.name}:")
            print(f"  Title: {parsed_paper.title}")
            print(f"  Sections found: {len(parsed_paper.sections)}")
            print(f"  References found: {len(parsed_paper.references)}")
            
            # Extract sentences
            sentences = extract_sentences(parsed_paper)
            extracted_sentences_dict[pdf_path.name] = sentences
            print(f"  Total sentences extracted: {len(sentences)}\n")
            
        except Exception as e:
            print(f"Error parsing {pdf_path.name}: {e}\n")

    # 2. Extract 5 example sentences from BERT and print markdown table
    print("--- 2. Thesis Visual: 5 Example Sentences Table ---")
    bert_filename = "BERT.pdf"
    if bert_filename in extracted_sentences_dict:
        bert_sentences = extracted_sentences_dict[bert_filename]
        # Try to select a mix of sentences that have citations and those that don't
        sample_sentences = []
        has_cite = [s for s in bert_sentences if s.has_citation]
        no_cite = [s for s in bert_sentences if not s.has_citation]
        
        sample_sentences.extend(has_cite[:3])
        sample_sentences.extend(no_cite[:2])
        
        df = pd.DataFrame([
            {
                "Section": s.section,
                "Position": f"{s.position_in_section:.2f}",
                "Has Citation": s.has_citation,
                "Retrieval Text (Snippet)": s.retrieval_text[:70] + "..." if len(s.retrieval_text) > 70 else s.retrieval_text
            }
            for s in sample_sentences
        ])
        
        print(df.to_markdown(index=False))
        print("\n")

        # 3. Bar chart: Sentences per section for BERT paper
        print("--- 3. Thesis Visual: Bar Chart of Sentences per Section ---")
        section_counts = {}
        for s in bert_sentences:
            section_counts[s.section] = section_counts.get(s.section, 0) + 1
            
        # Optional: Filter out tiny sections or sort
        filtered_counts = {k: v for k, v in section_counts.items() if v > 2}
        
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_counts.keys(), filtered_counts.values(), color='skyblue')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.title('Sentences per Section in BERT')
        plt.xlabel('Section Name')
        plt.ylabel('Sentence Count')
        plt.tight_layout()
        
        chart_path = assets_dir / "1_bert_sentences_per_section.png"
        plt.savefig(chart_path)
        print(f"Saved bar chart to {chart_path}")

if __name__ == "__main__":
    run_stage1_experiment()
