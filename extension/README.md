# Missing Citations Identifier — VS Code extension

Phase 4 of the citation-recommender MVP. Highlight a sentence in any document,
run **Recommend Citations for Selection**, pick a paper from the results, and
the selection is replaced with a citation marker in the right format for the
host file:

| Language       | Inserted marker          |
| -------------- | ------------------------ |
| LaTeX / TeX    | `\cite{key}`             |
| Markdown / Quarto | `[@key]`              |
| Anything else  | `(Surname et al., Year)` |

The extension is a thin client over the FastAPI backend (`POST /recommend`).
Backend, model, and DB live in the parent repo.

## Results picker

Each candidate is shown as a single row:

```
$(book) Devlin et al. (2019) — BERT: Pre-training of Deep…   ●●●●○ 78% match · NAACL · 4 authors
        ❝ …we use BERT to encode sentences… ❞  — cited 2020
```

- **Match meter** — the dot meter and `% match` come from the cosine
  similarity of the strongest supporting citation context, *not* the backend
  ranking score (which mixes in a corroboration bonus and is unbounded). The
  ranking score still controls row order.
- **Per-row buttons** (hover the row): **Copy BibTeX** drops the entry on the
  clipboard without closing the picker; **Search online** opens a Google
  Scholar search for the title.
- Type to filter — matching runs over the title, venue, and evidence too.

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
