# Missing Citations Identifier — VS Code extension

Phase 4 of the citation-recommender MVP. Highlight a sentence in any document,
run **Recommend Citations for Selection**, pick a paper from the QuickPick, and
the selection is replaced with a citation marker in the right format for the
host file:

| Language       | Inserted marker          |
| -------------- | ------------------------ |
| LaTeX / TeX    | `\cite{key}`             |
| Markdown / Quarto | `[@key]`              |
| Anything else  | `(Surname et al., Year)` |

The extension is a thin client over the FastAPI backend (`POST /recommend`).
Backend, model, and DB live in the parent repo.

## Configuration

| Setting                              | Default                   |
| ------------------------------------ | ------------------------- |
| `missingCitations.backendUrl`        | `http://localhost:8000`   |
| `missingCitations.topK`              | `10`                      |
| `missingCitations.requestTimeoutMs`  | `15000`                   |

## Develop

```bash
cd extension
npm install
npm run compile          # tsc → out/
code .
# Press F5 to launch the Extension Development Host
```

In the dev host, open a `.tex` or `.md` file, highlight a sentence such as
*"We use BERT to encode sentences."*, then run **Recommend Citations for
Selection** from the command palette or the editor context menu.

## Requirements

- VS Code 1.84+ (Node 20 runtime — needed for global `fetch`).
- The backend running locally (`uvicorn api.main:app --reload`) with embeddings
  and the HNSW index already built (Phase 3).
