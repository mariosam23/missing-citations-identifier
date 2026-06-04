"""Generate thesis-quality evaluation figures for the embedder fine-tuning
experiment (Phase 13).

Reads ``data/eval/reports/learning_curve.jsonl`` and
``data/finetune/simulated_pool.jsonl``, generates publication-ready
PDF + PNG figures suitable for LaTeX ``\\includegraphics``.

No GPU, no DB, no model required — pure matplotlib over JSON.

Usage::

    python -m scripts.finetune_evaluation
    python -m scripts.finetune_evaluation --output-dir data/eval/reports/figures
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.logger import logger

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Colorblind-safe academic palette (Okabe-Ito inspired, tuned for print+screen)
# ---------------------------------------------------------------------------
BLUE = "#2166ac"
ORANGE = "#e08214"
GREEN = "#1b7837"
RED = "#c51b7d"
PURPLE = "#762a83"
GRAY = "#636363"

METRIC_COLORS = {
    "Hit@1": BLUE,
    "Recall@20": ORANGE,
    "MRR@20": GREEN,
}

METRIC_MARKERS = {
    "Hit@1": "o",
    "Recall@20": "s",
    "MRR@20": "^",
}

# Metrics to extract from learning_curve.jsonl
KEY_METRICS = {
    "Hit@1": "val_cosine_accuracy@1",
    "Recall@20": "val_cosine_recall@20",
    "MRR@20": "val_cosine_mrr@20",
}

ALL_ACCURACY_KEYS = {
    "Accuracy@1": "val_cosine_accuracy@1",
    "Accuracy@5": "val_cosine_accuracy@5",
    "Accuracy@10": "val_cosine_accuracy@10",
    "Accuracy@20": "val_cosine_accuracy@20",
}

ALL_METRICS_KEYS = {
    "Hit@1": "val_cosine_accuracy@1",
    "Acc@5": "val_cosine_accuracy@5",
    "Acc@10": "val_cosine_accuracy@10",
    "Acc@20": "val_cosine_accuracy@20",
    "Prec@1": "val_cosine_precision@1",
    "Prec@5": "val_cosine_precision@5",
    "Prec@10": "val_cosine_precision@10",
    "Prec@20": "val_cosine_precision@20",
    "Rec@1": "val_cosine_recall@1",
    "Rec@5": "val_cosine_recall@5",
    "Rec@10": "val_cosine_recall@10",
    "Rec@20": "val_cosine_recall@20",
    "NDCG@10": "val_cosine_ndcg@10",
    "MRR@20": "val_cosine_mrr@20",
    "MAP@100": "val_cosine_map@100",
}

# ---------------------------------------------------------------------------
# Matplotlib style for thesis-quality output
# ---------------------------------------------------------------------------
RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2,
    "lines.markersize": 7,
}


def _apply_style() -> None:
    """Apply consistent thesis figure style."""
    for key, val in RCPARAMS.items():
        matplotlib.rcParams[key] = val


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as both PDF (LaTeX) and PNG (preview)."""
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    logger.info("Saved %s (.pdf + .png)", name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_learning_curve(path: Path) -> list[dict]:
    """Load and sort learning curve records by step number."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["step"])
    return records


def _load_pool(path: Path) -> list[dict]:
    """Load simulated triplet pool."""
    triplets: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


# ---------------------------------------------------------------------------
# Figure 1: Learning curve (multi-metric)
# ---------------------------------------------------------------------------

def plot_learning_curve(records: list[dict], output_dir: Path) -> None:
    """Multi-metric learning curve with baseline annotation."""
    steps = [r["step"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, key in KEY_METRICS.items():
        values = [r["metrics"][key] for r in records]
        color = METRIC_COLORS[label]
        marker = METRIC_MARKERS[label]
        ax.plot(
            steps, values,
            marker=marker, color=color, label=label,
            markeredgecolor="white", markeredgewidth=0.8,
        )
        # Baseline dashed line at Step 0
        ax.axhline(
            values[0], color=color, linestyle=":", alpha=0.5, linewidth=1,
        )

    # Annotate baseline
    ax.annotate(
        "Baseline\n(zero-shot)",
        xy=(0, records[0]["metrics"]["val_cosine_accuracy@1"]),
        xytext=(steps[-1] * 0.25, records[0]["metrics"]["val_cosine_accuracy@1"] + 0.035),
        fontsize=9, color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1),
    )

    ax.set_xlabel("Simulated feedback triplets")
    ax.set_ylabel("In-memory validation score")
    ax.set_title("Incremental Fine-Tuning Learning Curve")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(left=-50)

    _save_figure(fig, output_dir, "finetune_learning_curve")


# ---------------------------------------------------------------------------
# Figure 2: Metric delta bar chart (final − baseline)
# ---------------------------------------------------------------------------

def plot_metric_delta(records: list[dict], output_dir: Path) -> None:
    """Bar chart showing absolute change from baseline for each metric."""
    baseline = records[0]["metrics"]

    # Use the best step and the final step
    best_idx = max(
        range(1, len(records)),
        key=lambda i: records[i]["metrics"]["val_cosine_accuracy@1"],
    )
    best = records[best_idx]
    final = records[-1]

    metrics_to_show = {
        "Hit@1": "val_cosine_accuracy@1",
        "Acc@5": "val_cosine_accuracy@5",
        "Acc@10": "val_cosine_accuracy@10",
        "Acc@20": "val_cosine_accuracy@20",
        "Rec@20": "val_cosine_recall@20",
        "MRR@20": "val_cosine_mrr@20",
        "NDCG@10": "val_cosine_ndcg@10",
    }

    labels = list(metrics_to_show.keys())
    best_deltas = [
        best["metrics"][k] - baseline[k] for k in metrics_to_show.values()
    ]
    final_deltas = [
        final["metrics"][k] - baseline[k] for k in metrics_to_show.values()
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_best = ax.bar(
        x - width / 2, best_deltas, width,
        label=f"Best step ({best['step']} triplets)",
        color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    bars_final = ax.bar(
        x + width / 2, final_deltas, width,
        label=f"Final step ({final['step']} triplets)",
        color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.5,
    )

    # Color negative bars red-ish
    for bars in (bars_best, bars_final):
        for bar in bars:
            if bar.get_height() < 0:
                bar.set_edgecolor(RED)
                bar.set_linewidth(1.0)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Absolute change from baseline")
    ax.set_title("Metric Deltas vs. Zero-Shot Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="lower left", framealpha=0.9)

    _save_figure(fig, output_dir, "finetune_metric_delta")


# ---------------------------------------------------------------------------
# Figure 3: Normalised performance heatmap
# ---------------------------------------------------------------------------

def plot_performance_heatmap(records: list[dict], output_dir: Path) -> None:
    """Step × Metric heatmap, each metric normalised to its Step-0 value."""
    baseline = records[0]["metrics"]
    steps = [r["step"] for r in records]

    # Select a subset of metrics for readability
    metric_keys = {
        "Hit@1": "val_cosine_accuracy@1",
        "Acc@5": "val_cosine_accuracy@5",
        "Acc@10": "val_cosine_accuracy@10",
        "Acc@20": "val_cosine_accuracy@20",
        "Rec@20": "val_cosine_recall@20",
        "MRR@20": "val_cosine_mrr@20",
        "MAP@100": "val_cosine_map@100",
    }

    metric_names = list(metric_keys.keys())
    data = np.zeros((len(metric_names), len(steps)))

    for j, rec in enumerate(records):
        for i, (_, key) in enumerate(metric_keys.items()):
            base_val = baseline[key]
            if base_val > 0:
                data[i, j] = rec["metrics"][key] / base_val
            else:
                data[i, j] = 1.0

    # Custom diverging colormap: red (bad) → white (neutral) → green (good)
    cmap = LinearSegmentedColormap.from_list(
        "perf",
        [(0.0, "#c51b7d"), (0.5, "#f7f7f7"), (1.0, "#4d9221")],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    vmin = max(0.6, data.min() - 0.05)
    vmax = min(1.4, data.max() + 0.05)
    im = ax.imshow(
        data, aspect="auto", cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )

    # Annotate cells
    for i in range(len(metric_names)):
        for j in range(len(steps)):
            val = data[i, j]
            text_color = "white" if abs(val - 1.0) > 0.15 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold",
            )

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=9)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=9)
    ax.set_xlabel("Training triplets")
    ax.set_title("Normalised Performance (1.00 = baseline)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Ratio to baseline", fontsize=9)

    _save_figure(fig, output_dir, "finetune_performance_heatmap")


# ---------------------------------------------------------------------------
# Figure 4: Training pool statistics
# ---------------------------------------------------------------------------

def plot_pool_statistics(triplets: list[dict], output_dir: Path) -> None:
    """Distribution of triplets per cited_paper_id + summary stats."""
    paper_counts = Counter(t["cited_paper_id"] for t in triplets)
    counts = sorted(paper_counts.values(), reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: histogram of triplets per paper
    ax1.hist(
        counts, bins=30, color=BLUE, alpha=0.85,
        edgecolor="white", linewidth=0.5,
    )
    ax1.set_xlabel("Triplets per cited paper")
    ax1.set_ylabel("Number of papers")
    ax1.set_title("Training triplet distribution")
    ax1.axvline(
        np.median(counts), color=RED, linestyle="--",
        linewidth=1.5, label=f"Median = {np.median(counts):.0f}",
    )
    ax1.axvline(
        np.mean(counts), color=ORANGE, linestyle="-.",
        linewidth=1.5, label=f"Mean = {np.mean(counts):.1f}",
    )
    ax1.legend(fontsize=9)

    # Right: cumulative distribution (Lorenz-like)
    sorted_counts = np.sort(counts)
    cum_frac = np.cumsum(sorted_counts) / np.sum(sorted_counts)
    paper_frac = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax2.plot(paper_frac, cum_frac, color=BLUE, linewidth=2)
    ax2.plot([0, 1], [0, 1], color=GRAY, linestyle=":", linewidth=1)
    ax2.set_xlabel("Fraction of cited papers")
    ax2.set_ylabel("Cumulative fraction of triplets")
    ax2.set_title("Training pool concentration")
    ax2.fill_between(paper_frac, cum_frac, paper_frac, alpha=0.1, color=BLUE)

    # Stats annotation
    stats_text = (
        f"Total triplets: {len(triplets):,}\n"
        f"Distinct papers: {len(paper_counts):,}\n"
        f"Max per paper: {max(counts)}\n"
        f"Cap: 50/paper"
    )
    ax2.text(
        0.05, 0.95, stats_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    _save_figure(fig, output_dir, "finetune_pool_statistics")


# ---------------------------------------------------------------------------
# Figure 5: Accuracy@k grouped bar comparison
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(records: list[dict], output_dir: Path) -> None:
    """Grouped bar chart: baseline vs best vs final across all k values."""
    baseline = records[0]

    # Find best fine-tuned step (by Hit@1)
    best_idx = max(
        range(1, len(records)),
        key=lambda i: records[i]["metrics"]["val_cosine_accuracy@1"],
    )
    best = records[best_idx]
    final = records[-1]

    k_labels = list(ALL_ACCURACY_KEYS.keys())
    k_keys = list(ALL_ACCURACY_KEYS.values())

    baseline_vals = [baseline["metrics"][k] for k in k_keys]
    best_vals = [best["metrics"][k] for k in k_keys]
    final_vals = [final["metrics"][k] for k in k_keys]

    x = np.arange(len(k_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(
        x - width, baseline_vals, width,
        label="Baseline (zero-shot)", color=BLUE,
        alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x, best_vals, width,
        label=f"Best step ({best['step']})",
        color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x + width, final_vals, width,
        label=f"Final step ({final['step']})",
        color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.5,
    )

    ax.set_ylabel("Accuracy (in-memory evaluator)")
    ax.set_title("Retrieval Accuracy at Different Depths")
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.legend(loc="upper left", framealpha=0.9)

    # Add value labels on bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    _save_figure(fig, output_dir, "finetune_accuracy_comparison")


# ---------------------------------------------------------------------------
# Figure 6: Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(records: list[dict], output_dir: Path) -> None:
    """Matplotlib table with all metrics across all steps."""
    steps = [r["step"] for r in records]

    # Select key metrics for the table
    table_metrics = {
        "Hit@1": "val_cosine_accuracy@1",
        "Acc@5": "val_cosine_accuracy@5",
        "Acc@10": "val_cosine_accuracy@10",
        "Acc@20": "val_cosine_accuracy@20",
        "Prec@20": "val_cosine_precision@20",
        "Rec@20": "val_cosine_recall@20",
        "NDCG@10": "val_cosine_ndcg@10",
        "MRR@20": "val_cosine_mrr@20",
        "MAP@100": "val_cosine_map@100",
    }

    metric_names = list(table_metrics.keys())

    # Build table data
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    baseline_vals = {
        k: records[0]["metrics"][v] for k, v in table_metrics.items()
    }

    for rec in records:
        row: list[str] = []
        row_colors: list[str] = []
        for mname, mkey in table_metrics.items():
            val = rec["metrics"][mkey]
            base = baseline_vals[mname]
            row.append(f"{val:.4f}")
            if rec["step"] == 0:
                row_colors.append("#f0f0f0")
            elif val >= base:
                row_colors.append("#e6f5e6")  # light green
            else:
                row_colors.append("#fce4ec")  # light red
        cell_text.append(row)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(
        figsize=(10, 0.5 + 0.35 * len(records)),
    )
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=[str(s) for s in steps],
        colLabels=metric_names,
        cellColours=cell_colors,
        rowColours=["#e8e8e8"] * len(steps),
        colColours=[BLUE + "30"] * len(metric_names),
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=8)
        if col == -1:
            cell.set_text_props(fontweight="bold", fontsize=9)

    ax.set_title(
        "In-Memory Evaluator Metrics Across Training Steps",
        fontsize=12, fontweight="bold", pad=20,
    )

    # Add note about Step 0
    fig.text(
        0.5, 0.02,
        "Step 0 = zero-shot baseline (BGE-Large, no fine-tuning). "
        "Green = ≥ baseline, Red = < baseline.",
        ha="center", fontsize=8, color=GRAY,
    )

    _save_figure(fig, output_dir, "finetune_summary_table")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    curve_path: str = typer.Option(
        "data/eval/reports/learning_curve.jsonl",
        help="Path to learning_curve.jsonl",
    ),
    pool_path: str = typer.Option(
        "data/finetune/simulated_pool.jsonl",
        help="Path to simulated_pool.jsonl",
    ),
    output_dir: str = typer.Option(
        "data/eval/reports/figures",
        help="Output directory for figures",
    ),
) -> None:
    """Generate thesis-quality evaluation figures for Phase 13."""
    _apply_style()

    curve_file = Path(curve_path)
    pool_file = Path(pool_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not curve_file.exists():
        logger.error("Learning curve file not found: %s", curve_file)
        raise typer.Exit(code=1)

    records = _load_learning_curve(curve_file)
    logger.info(
        "Loaded %d learning curve data points (steps: %s)",
        len(records),
        [r["step"] for r in records],
    )

    # Figure 1: Learning curve
    logger.info("Generating learning curve plot...")
    plot_learning_curve(records, out)

    # Figure 2: Metric delta
    if len(records) >= 2:
        logger.info("Generating metric delta chart...")
        plot_metric_delta(records, out)

    # Figure 3: Normalised heatmap
    if len(records) >= 2:
        logger.info("Generating performance heatmap...")
        plot_performance_heatmap(records, out)

    # Figure 4: Pool statistics
    if pool_file.exists():
        logger.info("Generating pool statistics...")
        triplets = _load_pool(pool_file)
        plot_pool_statistics(triplets, out)
    else:
        logger.warning("Pool file not found: %s — skipping pool stats", pool_file)

    # Figure 5: Accuracy@k comparison
    if len(records) >= 2:
        logger.info("Generating accuracy comparison...")
        plot_accuracy_comparison(records, out)

    # Figure 6: Summary table
    logger.info("Generating summary table...")
    plot_summary_table(records, out)

    logger.info("All figures saved to %s", out)


if __name__ == "__main__":
    app()
