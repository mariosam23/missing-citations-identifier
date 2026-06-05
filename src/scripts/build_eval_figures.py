"""Render thesis-ready figures for the missing-citation detection phase.

Reads the JSON evaluation reports already produced by the eval scripts and writes
PNGs to ``data/eval/figures/`` plus a plot-only ``notebooks/eval_figures.ipynb``
that embeds them. Each figure is generated independently: a missing input report
is logged and skipped, never fatal.

Run from the repo root::

    python -m scripts.build_eval_figures
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat as nbf  # noqa: E402
from matplotlib.container import BarContainer  # noqa: E402

from utils.logger import logger  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _PROJECT_ROOT / "data" / "eval" / "reports"
_FIGURES_DIR = _PROJECT_ROOT / "data" / "eval" / "figures"
_NOTEBOOK_PATH = _PROJECT_ROOT / "notebooks" / "eval_figures.ipynb"

_DPI = 150
_RULE_COLOR = "#7f7f7f"
_LLM_COLOR = "#1f77b4"
_FILTER_COLOR = "#2ca02c"
_ACCENT = "#ff7f0e"

# (filename, title, one-line caption) for every figure the notebook embeds.
_FIGURE_INDEX: list[tuple[str, str, str]] = []


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning("report not found, skipping figure: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str, title: str, caption: str) -> None:
    out = _FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    _FIGURE_INDEX.append((out.name, title, caption))
    logger.info("wrote %s", out)


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt=fmt, padding=2, fontsize=8)


def fig_rule_confusion() -> None:
    report = _load(_REPORTS_DIR / "missing_manual_tei.json")
    if report is None:
        return
    matrix = report["confusion_matrix"]
    labels = ["HAS_CITATION", "MISSING_CITATION", "NOT_CITATION_WORTHY"]
    short = ["HAS", "MISSING", "NOT-WORTHY"]
    data = [[matrix[g][p] for p in labels] for g in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)), short)
    ax.set_yticks(range(len(labels)), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Rule detector — confusion matrix (1,200 balanced sentences)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = data[i][j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > max(max(r) for r in data) / 2 else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(
        fig,
        "01_rule_confusion_matrix",
        "Rule detector confusion matrix",
        "The rule detector almost never flags MISSING_CITATION (recall ~0.05) but "
        "never re-flags an already-cited sentence.",
    )


def fig_llm_confusion() -> None:
    report = _load(_REPORTS_DIR / "citation_need_binary.json")
    if report is None:
        return
    h = report["headline"]
    data = [[h["true_positive"], h["false_negative"]],
            [h["false_positive"], h["true_negative"]]]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(data, cmap="Greens")
    ax.set_xticks([0, 1], ["pred: needs cite", "pred: no cite"])
    ax.set_yticks([0, 1], ["gold: needs cite", "gold: no cite"])
    ax.set_title("LLM identifier — binary confusion matrix")
    flat_max = max(max(r) for r in data)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(data[i][j]),
                ha="center",
                va="center",
                color="white" if data[i][j] > flat_max / 2 else "black",
                fontsize=14,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(
        fig,
        "02_llm_confusion_matrix",
        "LLM identifier confusion matrix",
        "The LLM recovers most should-cite sentences; remaining false positives "
        "are largely weak-label noise.",
    )


def fig_rule_vs_llm() -> None:
    rule = _load(_REPORTS_DIR / "missing_manual_tei.json")
    llm = _load(_REPORTS_DIR / "citation_need_binary.json")
    if rule is None or llm is None:
        return
    metrics = ["precision", "recall", "f1"]
    rule_vals = [rule["missing_citation"][m] for m in metrics]
    llm_vals = [llm["headline"][m] for m in metrics]

    x = range(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], rule_vals, width,
           label="Rule detector", color=_RULE_COLOR)
    ax.bar([i + width / 2 for i in x], llm_vals, width,
           label="LLM identifier", color=_LLM_COLOR)
    ax.set_xticks(list(x), [m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Should-cite identification: rule detector vs LLM")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _annotate_bars(ax)
    _save(
        fig,
        "03_rule_vs_llm",
        "Rule vs LLM identification",
        "Switching the decision to the LLM lifts should-cite recall from 0.05 to "
        "0.84 (~17x) at comparable precision.",
    )


def fig_threshold_sweep() -> None:
    report = _load(_REPORTS_DIR / "citation_need_binary.json")
    if report is None:
        return
    by_threshold = report["by_threshold"]
    thresholds = sorted(float(t) for t in by_threshold)
    keys = [f"{t:.2f}" for t in thresholds]
    recall = [by_threshold[k]["recall"] for k in keys]
    precision = [by_threshold[k]["precision"] for k in keys]
    fpr = [by_threshold[k]["false_positive_rate"] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, recall, marker="o", label="Recall", color=_LLM_COLOR)
    ax.plot(thresholds, precision, marker="s", label="Precision", color=_FILTER_COLOR)
    ax.plot(thresholds, fpr, marker="^", label="False-positive rate", color=_ACCENT)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("LLM identifier — self-reported confidence sweep")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(
        fig,
        "04_threshold_sweep",
        "Confidence threshold sweep",
        "Self-reported confidence is a weak knob: metrics are flat until ~0.9, "
        "where a small precision/recall trade appears.",
    )


def fig_filter_ablation() -> None:
    # Show every available run so free-tier model-rotation variance is visible:
    # two filter-OFF runs plus the filter-ON run. A clean two-bar A/B would imply
    # a controlled result the rotated runs cannot support.
    candidates = [
        ("Off (run A, cached)", _RULE_COLOR, "citation_need_binary.json"),
        ("Off (run B, session)", "#bcbd22", "citation_need_nofilter.json"),
        ("On (input filter)", _FILTER_COLOR, "citation_need_filtered.json"),
    ]
    runs: list[tuple[str, str, dict[str, Any]]] = []
    for label, color, fname in candidates:
        rep = _load(_REPORTS_DIR / fname)
        if rep is not None:
            runs.append((label, color, rep))
    if not any(label.startswith("On") for label, _c, _r in runs) or len(runs) < 2:
        logger.warning(
            "filter A/B figure skipped — need the filtered run plus at least one "
            "no-filter run (scripts.evaluate_citation_need run --filter/--no-filter)"
        )
        return

    metrics = ["precision", "recall", "f1"]
    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(metrics))
    n = len(runs)
    width = 0.8 / n
    for offset, (label, color, rep) in enumerate(runs):
        vals = [rep["headline"][m] for m in metrics]
        positions = [i + (offset - (n - 1) / 2) * width for i in x]
        ax_m.bar(positions, vals, width, label=label, color=color)
    ax_m.set_xticks(list(x), [m.capitalize() for m in metrics])
    ax_m.set_ylim(0, 1.05)
    ax_m.set_ylabel("Score")
    ax_m.set_title("Should-cite metrics across runs (rotation variance shown)")
    ax_m.legend(fontsize=8)
    ax_m.grid(axis="y", linestyle="--", alpha=0.5)
    _annotate_bars(ax_m)

    labels = [label for label, _c, _r in runs]
    colors = [color for _l, color, _r in runs]
    fprs = [rep["headline"]["false_positive_rate"] for _l, _c, rep in runs]
    ax_f.bar(range(len(runs)), fprs, color=colors)
    ax_f.set_xticks(range(len(runs)), labels, rotation=15, ha="right", fontsize=8)
    ax_f.set_ylim(0, 1.0)
    ax_f.set_ylabel("False-positive rate (lower is better)")
    ax_f.set_title("False-positive rate")
    ax_f.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(fprs):
        ax_f.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    _save(
        fig,
        "05_filter_ablation",
        "Input-sanitization filter — runs comparison",
        "Two filter-OFF runs differ by ~0.16 recall, so free-tier model-rotation "
        "non-determinism dominates: the filter's headline-metric effect on this "
        "clean eval set is within that noise. Its deterministic value is residue "
        "removal, label-leak removal, and skipping non-prose targets.",
    )


def fig_end_to_end() -> None:
    synth = _load(_REPORTS_DIR / "missing_synthetic.json")
    llm = _load(_REPORTS_DIR / "citation_need_binary.json")
    if synth is None or llm is None:
        return
    rec_on_detected = synth["recommendation_recall_on_detected"]["recall@20"]
    rule_detect = synth["detector_recall"]
    rule_e2e = synth["end_to_end_recall"]["recall@20"]
    llm_detect = llm["headline"]["recall"]
    llm_e2e = llm_detect * rec_on_detected  # projected

    groups = ["Detector recall", "End-to-end recall@20"]
    rule_vals = [rule_detect, rule_e2e]
    llm_vals = [llm_detect, llm_e2e]

    x = range(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar([i - width / 2 for i in x], rule_vals, width,
           label="Rule detector pipeline", color=_RULE_COLOR)
    ax.bar([i + width / 2 for i in x], llm_vals, width,
           label="LLM identifier pipeline (projected)", color=_LLM_COLOR)
    ax.set_xticks(list(x), groups)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_title(
        "End-to-end identify→recommend "
        f"(recommender@20 on detected = {rec_on_detected:.2f})"
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _annotate_bars(ax)
    _save(
        fig,
        "06_end_to_end",
        "End-to-end staged recall",
        "End-to-end recall = detector recall x recommender-on-detected. The LLM "
        "bar is projected from the measured LLM recall and the recommender's "
        "recall on detected sentences.",
    )


def fig_density_stratified() -> None:
    report = _load(_REPORTS_DIR / "stratified_dense_only_test_2026-05-29.json")
    if report is None:
        return
    buckets = report["buckets"]
    order = [b for b in ("2", "3-5", "6-10", "10+") if b in buckets]
    ks = ["recall@5", "recall@10", "recall@20"]
    colors = ["#9ecae1", _LLM_COLOR, "#08519c"]

    x = range(len(order))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, (k, color) in enumerate(zip(ks, colors, strict=True)):
        vals = [buckets[b][k] for b in order]
        ax.bar([i + (offset - 1) * width for i in x], vals, width,
               label=k, color=color)
    ax.set_xticks(list(x), [f"{b} citers" for b in order])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Gold paper's distinct-citer count (corpus density)")
    ax.set_title("Recommendation recall rises with citation density (test split)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save(
        fig,
        "07_density_stratified",
        "Recall by citation density",
        "The recommender's recall climbs monotonically with how many corpus "
        "papers cite the gold paper — corpus density is the ceiling.",
    )


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Missing-citation detection — evaluation figures\n\n"
            "Plot-only notebook. Figures are generated by "
            "`python -m scripts.build_eval_figures` and embedded from "
            "`../data/eval/figures/`. Re-run that script to refresh them."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n\n"
            "FIG_DIR = Path('../data/eval/figures')"
        ),
    ]
    for name, title, caption in _FIGURE_INDEX:
        cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{caption}"))
        cells.append(
            nbf.v4.new_code_cell(f"display(Image(filename=str(FIG_DIR / '{name}')))")
        )
    nb["cells"] = cells
    _NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)
    logger.info("wrote %s", _NOTEBOOK_PATH)


def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[Callable[[], None]] = [
        fig_rule_confusion,
        fig_llm_confusion,
        fig_rule_vs_llm,
        fig_threshold_sweep,
        fig_filter_ablation,
        fig_end_to_end,
        fig_density_stratified,
    ]
    for figure in figures:
        try:
            figure()
        except Exception:  # noqa: BLE001 - one bad figure must not kill the rest
            logger.exception("figure %s failed", figure.__name__)
    build_notebook()
    logger.info(
        "done: %d figure(s) in %s", len(_FIGURE_INDEX), _FIGURES_DIR
    )


if __name__ == "__main__":
    main()
