"""Generate the Colab notebook that runs the incremental fine-tune learning curve.

Companion to ``src/scripts/incremental_finetune.py``. The fine-tune reloads
bge-large ~10x and re-encodes the in-memory eval corpus on every step, which
saturates a 6 GB laptop GPU for hours — so we run it on a Colab T4 instead,
against the local Postgres exposed over an ngrok TCP tunnel (the same pattern as
``notebooks/colab_embed_contexts.ipynb``).

Run ``python -m scripts.build_colab_learning_curve_notebook`` (or
``python src/scripts/build_colab_learning_curve_notebook.py``) to (re)write
``notebooks/colab_learning_curve.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path("notebooks/colab_learning_curve.ipynb")

# --- Cell sources -----------------------------------------------------------

_INTRO_MD = """\
# Incremental fine-tune learning curve on Colab GPU

Runs `scripts.incremental_finetune` on a Colab T4 against your local Postgres
exposed via `ngrok tcp 25432`. As simulated feedback triplets accumulate, the
notebook fine-tunes bge-large in chunks and evaluates each step with an
in-memory `InformationRetrievalEvaluator`, writing one row per step to
`learning_curve.jsonl`.

**Before running**
1. `Runtime → Change runtime type → T4 GPU`.
2. On your laptop, start the tunnel: `ngrok tcp 25432` (or whatever port your
   `papers_db` container maps to) and keep it open.
3. **Commit and push** the `new_implementation` branch first — including the
   query-prompt fix in `incremental_finetune.py`. The clone below pulls from
   GitHub, so an uncommitted local edit will not be picked up.
4. Have `split_42.json` and `simulated_pool.jsonl` ready to upload (both are
   under the gitignored `data/`, so they are not in the clone).
"""

_CONN_MD = """\
## 1. Connection details

Paste the host/port from the `ngrok tcp` window and your Postgres credentials
(the ones in `.env` / `colab_embed_contexts.ipynb`). Then set the training
hyperparameters — `BATCH_SIZE = 16` is comfortable on a 16 GB T4 (triplets =
3 forward passes per step); raise `CHUNK_SIZE` to evaluate at coarser steps.
"""

_CONN_CODE = """\
NGROK_HOST = "0.tcp.eu.ngrok.io"   # <-- replace from the ngrok window
NGROK_PORT = 00000                  # <-- replace from the ngrok window
POSTGRES_USER = "<your-db-user>"        # match your .env
POSTGRES_PASSWORD = "<your-db-password>"  # match your .env
POSTGRES_DB = "papers_db"               # match your .env

REPO_URL = "https://github.com/mariosam23/missing-citations-identifier"
BRANCH = "new_implementation"

# Training hyperparameters (T4-sized).
BATCH_SIZE = 16          # per-device; 3 fwd passes per triplet
CHUNK_SIZE = 500         # triplets added per evaluated step
GRAD_ACCUM_STEPS = 4     # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS

import os
os.environ["DB_URL"] = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{NGROK_HOST}:{NGROK_PORT}/{POSTGRES_DB}"
)
os.environ["EMBEDDER_MODEL_NAME"] = "BAAI/bge-large-en-v1.5"
os.environ["EMBEDDER_DIM"] = "1024"
os.environ["EMBEDDER_DEVICE"] = "cuda"
print("DB target:", os.environ["DB_URL"].split("@")[1])
"""

_INSTALL_MD = """\
## 2. Clone repo and install dependencies

Colab ships Python 3.11; the project pins 3.13, so we install the runtime deps
directly rather than `pip install -e .`. Nothing in the fine-tune path uses
3.13-only syntax.
"""

_INSTALL_CODE = """\
!git clone --depth 1 --branch {BRANCH} {REPO_URL} /content/repo
%cd /content/repo
!pip install -q --upgrade pip
!pip install -q \\
    'sentence-transformers>=3.2' 'torch>=2.4' \\
    'sqlalchemy>=2.0' 'psycopg[binary]>=3.2' 'pgvector>=0.3.6' \\
    'pydantic>=2.7' 'pydantic-settings>=2.4' \\
    'typer>=0.12' 'tqdm>=4.66' matplotlib xformers einops
import torch
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
"""

_UPLOAD_MD = """\
## 3. Upload the gitignored data files

`split_42.json` and `simulated_pool.jsonl` live under the gitignored `data/`
directory, so they are not in the clone. Upload both here (the file picker
accepts a multi-select). The pool is ~3 MB.
"""

_UPLOAD_CODE = """\
import os
from google.colab import files

os.makedirs("/content/repo/data/eval", exist_ok=True)
os.makedirs("/content/repo/data/finetune", exist_ok=True)

print("Select BOTH split_42.json and simulated_pool.jsonl ...")
uploaded = files.upload()
for name, content in uploaded.items():
    if "split" in name:
        dest = "/content/repo/data/eval/split_42.json"
    elif "pool" in name or name.endswith(".jsonl"):
        dest = "/content/repo/data/finetune/simulated_pool.jsonl"
    else:
        dest = f"/content/repo/{name}"
    with open(dest, "wb") as fh:
        fh.write(content)
    print("wrote", dest, f"({len(content)} bytes)")
"""

_SMOKE_MD = """\
## 4. Smoke test: DB reachable + embedder loads

Confirms the ngrok tunnel works and bge-large loads with the asymmetric query
prompt injected (the fix this run validates). Frees the model afterwards so the
training subprocess starts with a clean GPU.
"""

_SMOKE_CODE = """\
import sys
sys.path.insert(0, "/content/repo/src")
import numpy as np
from sqlalchemy import text
from database.postgres.engine import get_session
from pipeline.embedding.embedder import encode_texts, get_embedder

with get_session() as session:
    n = session.execute(
        text("SELECT COUNT(*) FROM citation_contexts WHERE cited_paper_id IS NOT NULL")
    ).scalar_one()
print("resolved citation_contexts reachable over tunnel:", n)

vectors = encode_texts(["The transformer relies on self-attention."], is_query=True)
print("query vec dim:", vectors.shape[1], "L2:", float(np.linalg.norm(vectors[0])))

import gc, torch
import pipeline.embedding.embedder as _emb
_emb._model = None
gc.collect()
torch.cuda.empty_cache()
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
"""

_RUN_MD = """\
## 5. Run the learning curve

Resumable: each completed step is appended to `learning_curve.jsonl`, and a
re-run skips steps already present. If Colab disconnects mid-run, just re-run
this cell. `expandable_segments` reduces fragmentation across the repeated model
reloads. Expect a couple of hours (the in-memory eval re-encodes the corpus each
step).
"""

_RUN_CODE = """\
!cd /content/repo && PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \\
    python -m scripts.incremental_finetune \\
        --pool-path data/finetune/simulated_pool.jsonl \\
        --split-path data/eval/split_42.json \\
        --output-path data/eval/reports/learning_curve.jsonl \\
        --chunk-size {CHUNK_SIZE} \\
        --batch-size {BATCH_SIZE} \\
        --grad-accum-steps {GRAD_ACCUM_STEPS}
"""

_PLOT_MD = """\
## 6. Plot the curve

Step 0 is the zero-shot bge-large baseline (now with the query prompt, so it is
comparable to production). A flat-or-declining curve confirms the
false-negative-mining hypothesis; an upward curve is the positive result.
"""

_PLOT_CODE = """\
import json
import matplotlib.pyplot as plt

steps, hit1, recall20, mrr20 = [], [], [], []
with open("/content/repo/data/eval/reports/learning_curve.jsonl") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        m = rec["metrics"]
        steps.append(rec["step"])
        hit1.append(m.get("val_cosine_accuracy@1", 0.0))
        recall20.append(m.get("val_cosine_recall@20", 0.0))
        mrr20.append(m.get("val_cosine_mrr@20", 0.0))

order = sorted(range(len(steps)), key=lambda i: steps[i])
steps = [steps[i] for i in order]
hit1 = [hit1[i] for i in order]
recall20 = [recall20[i] for i in order]
mrr20 = [mrr20[i] for i in order]

plt.figure(figsize=(10, 6))
plt.plot(steps, hit1, marker="o", label="Hit@1", linewidth=2)
plt.plot(steps, recall20, marker="s", label="Recall@20", linewidth=2)
plt.plot(steps, mrr20, marker="^", label="MRR@20", linewidth=2)
plt.title("Incremental fine-tuning learning curve", fontsize=14, fontweight="bold")
plt.xlabel("Simulated feedback triplets")
plt.ylabel("Validation score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("/content/repo/data/eval/reports/learning_curve.png", dpi=150)
plt.show()
"""

_DOWNLOAD_MD = """\
## 7. Download the results

Pull the curve (and plot) back to your laptop to commit into `data/eval/reports`
and `thesis/`. The fine-tuned model is saved under `models/` on Colab's
ephemeral disk — download it separately only if you intend to deploy it.
"""

_DOWNLOAD_CODE = """\
from google.colab import files
files.download("/content/repo/data/eval/reports/learning_curve.jsonl")
files.download("/content/repo/data/eval/reports/learning_curve.png")
"""


def build() -> nbf.NotebookNode:
    """Assemble the notebook node."""
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(_INTRO_MD),
        nbf.v4.new_markdown_cell(_CONN_MD),
        nbf.v4.new_code_cell(_CONN_CODE),
        nbf.v4.new_markdown_cell(_INSTALL_MD),
        nbf.v4.new_code_cell(_INSTALL_CODE),
        nbf.v4.new_markdown_cell(_UPLOAD_MD),
        nbf.v4.new_code_cell(_UPLOAD_CODE),
        nbf.v4.new_markdown_cell(_SMOKE_MD),
        nbf.v4.new_code_cell(_SMOKE_CODE),
        nbf.v4.new_markdown_cell(_RUN_MD),
        nbf.v4.new_code_cell(_RUN_CODE),
        nbf.v4.new_markdown_cell(_PLOT_MD),
        nbf.v4.new_code_cell(_PLOT_CODE),
        nbf.v4.new_markdown_cell(_DOWNLOAD_MD),
        nbf.v4.new_code_cell(_DOWNLOAD_CODE),
    ]
    nb["metadata"] = {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"Generated {NOTEBOOK_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
