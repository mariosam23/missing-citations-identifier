# Novel-Work Eval Dataset

A targeted dataset for measuring whether the **Contribution Profile** classifier upgrade (see [src/pipeline/contribution_profile.py](../../src/pipeline/contribution_profile.py)) reduces false positives where the classifier flags the authors' own novel work as `MISSING_CITATION`.

## Schema (`dataset.jsonl`)

One JSON object per line:

| Field | Description |
|---|---|
| `paper_id` | Stable id, lowercase. Used to look up the PDF and cache the profile. |
| `sentence_id` | `{paper_id}::{section_or_subid}::{seq}`. |
| `text` | The sentence as it would appear in `SentenceRecord.text` (citation markers preserved). |
| `section` | Section name as it appears in the parsed paper. |
| `previous_sentence` / `next_sentence` | Immediate neighbours; `null` if absent. |
| `gold_label` | One of `NOT_CITATION_WORTHY`, `MISSING_CITATION`, `COVERED_BY_BLOCK`, `HAS_CITATION`. |
| `rationale` | One short sentence explaining the label. Helps reviewers audit. |

## Composition guidance

- Bias toward `gold_label = NOT_CITATION_WORTHY` (the priority failure mode).
- Include adversarial controls (~25%): sentences that surround novel-work descriptions but actually credit a `uses`-list item (e.g. "Transformer", "WordPiece") and SHOULD be `MISSING_CITATION`. These guard against over-suppression by the new rule 8.
- Include sentences that DO have a citation marker — those should stay `HAS_CITATION` regardless of the profile.

## Adding a new paper

1. Drop the PDF under `papers/<NAME>.pdf`.
2. Add an entry to `DEFAULT_PAPER_PATHS` in [run_eval.py](run_eval.py) (or pass `--paper id=path`).
3. Add labelled rows to `dataset.jsonl`.
4. Run `python eval/novel_work_dataset/run_eval.py` — the profile is extracted once and cached at `profiles/{paper_id}.json`.

## Running

```bash
python eval/novel_work_dataset/run_eval.py
# overrides:
python eval/novel_work_dataset/run_eval.py --paper bert=papers/BERT.pdf --delay 12
# only inspect augmented behavior (skips baseline run):
python eval/novel_work_dataset/run_eval.py --skip-baseline
```

The runner reports:
- Per-class precision/recall under baseline vs augmented prompt.
- Per-sentence flips, tagged `CORRECTED` / `REGRESSED` / `CHANGED`.
- Confusion matrices.

## Editing cached profiles by hand

Profiles cache under `profiles/{paper_id}.json`. If the LLM put a known prior-work concept into `proposes`, edit the JSON directly and re-run — the cache is honoured.
