"""Evaluate the contribution-profile classifier upgrade.

Loads ``eval/novel_work_dataset/dataset.jsonl``, runs the classifier twice on
the same sentences — once with the base system prompt, once with the augmented
prompt that injects the paper's contribution profile — and reports how many
sentences flipped, plus precision/recall on the NOT_CITATION_WORTHY class
(the priority metric for avoiding false positives).

Usage:
    python eval/novel_work_dataset/run_eval.py
    python eval/novel_work_dataset/run_eval.py --paper bert=papers/BERT.pdf

Profiles are cached under ``eval/novel_work_dataset/profiles/{paper_id}.json``
so repeat runs do not re-call the LLM extractor.
"""


from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Make src/ importable when invoked from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from entities import ContributionProfile, SentenceRecord  # noqa: E402
from pipeline.classifier import CitationClassifier  # noqa: E402
from pipeline.contribution_profile import (  # noqa: E402
    ContributionProfileExtractor,
    build_classifier_system_prompt,
)
from pipeline.pdf_parser import GrobidPDFParser  # noqa: E402
from utils.config import config  # noqa: E402


DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
PROFILES_DIR = Path(__file__).parent / "profiles"

DEFAULT_PAPER_PATHS: dict[str, Path] = {
    "bert": _REPO_ROOT / "papers" / "BERT.pdf",
}


@dataclass
class Example:
    paper_id: str
    sentence_id: str
    text: str
    section: str
    previous_sentence: str | None
    next_sentence: str | None
    gold_label: str
    rationale: str


def load_dataset(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    paper_id=row["paper_id"],
                    sentence_id=row["sentence_id"],
                    text=row["text"],
                    section=row.get("section", ""),
                    previous_sentence=row.get("previous_sentence"),
                    next_sentence=row.get("next_sentence"),
                    gold_label=row["gold_label"],
                    rationale=row.get("rationale", ""),
                )
            )
    return examples


def load_or_build_profile(paper_id: str, paper_path: Path) -> ContributionProfile:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    cached = PROFILES_DIR / f"{paper_id}.json"
    if cached.exists():
        return ContributionProfile.from_dict(json.loads(cached.read_text(encoding="utf-8")))

    if not paper_path.exists():
        raise FileNotFoundError(
            f"PDF not found for paper_id={paper_id!r} at {paper_path}. "
            f"Pass --paper {paper_id}=<path> or place the PDF at the default location."
        )
    print(f"[profile] parsing {paper_path} via GROBID...")
    parsed = GrobidPDFParser(str(paper_path)).parse()
    extractor = ContributionProfileExtractor(model=config.CLASSIFIER_BACKUP[1])
    profile = extractor.extract(parsed)
    cached.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    print(f"[profile] cached at {cached}")
    return profile


def to_sentence_record(ex: Example) -> SentenceRecord:
    return SentenceRecord(
        text=ex.text,
        section=ex.section,
        position_in_section=0.5,
        has_citation=False,
        retrieval_text=ex.text,
        previous_sentence=ex.previous_sentence,
        next_sentence=ex.next_sentence,
    )


def classify_with_prompt(
    sentences: list[SentenceRecord],
    paper_title: str,
    paper_abstract: str,
    system_prompt: str | None,
    batch_size: int,
    delay: float,
) -> list[str]:
    classifier = CitationClassifier(
        model=config.CLASSIFIER_BACKUP[1],
        batch_size=batch_size,
        delay_between_calls_seconds=delay,
        system_prompt=system_prompt,
    )
    classified = classifier.classify_sentences(sentences, paper_title, paper_abstract)
    return [c.citation_state.name if c.citation_state else "NONE" for c in classified]


def precision_recall_for(label: str, gold: list[str], pred: list[str]) -> tuple[float, float, int, int, int]:
    tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn


def confusion(gold: list[str], pred: list[str]) -> dict[tuple[str, str], int]:
    c: Counter = Counter()
    for g, p in zip(gold, pred):
        c[(g, p)] += 1
    return dict(c)


def report(
    examples: list[Example],
    pred_baseline: list[str],
    pred_augmented: list[str],
) -> None:
    gold = [ex.gold_label for ex in examples]
    n = len(gold)

    print("\n" + "=" * 72)
    print(f"Evaluated {n} examples")
    print("=" * 72)

    for label in ("NOT_CITATION_WORTHY", "MISSING_CITATION", "COVERED_BY_BLOCK", "HAS_CITATION"):
        if label not in gold:
            continue
        p_b, r_b, tp_b, fp_b, fn_b = precision_recall_for(label, gold, pred_baseline)
        p_a, r_a, tp_a, fp_a, fn_a = precision_recall_for(label, gold, pred_augmented)
        print(f"\n[{label}]   gold count = {gold.count(label)}")
        print(f"  baseline:  P={p_b:.3f}  R={r_b:.3f}   tp={tp_b} fp={fp_b} fn={fn_b}")
        print(f"  augmented: P={p_a:.3f}  R={r_a:.3f}   tp={tp_a} fp={fp_a} fn={fn_a}")
        print(f"  delta:     dP={p_a - p_b:+.3f}  dR={r_a - r_b:+.3f}")

    print("\n--- Per-sentence flips (baseline -> augmented) ---")
    flips = 0
    for ex, b, a in zip(examples, pred_baseline, pred_augmented):
        if b == a:
            continue
        flips += 1
        verdict = "CORRECTED" if a == ex.gold_label else ("REGRESSED" if b == ex.gold_label else "CHANGED")
        print(f"  [{verdict}] {ex.sentence_id}  gold={ex.gold_label}  {b} -> {a}")
        print(f"     text: {ex.text[:120]}{'...' if len(ex.text) > 120 else ''}")
    if flips == 0:
        print("  (no flips)")

    print(f"\nTotal flips: {flips}/{n}")
    print("\nBaseline confusion (gold, pred):", confusion(gold, pred_baseline))
    print("Augmented confusion (gold, pred):", confusion(gold, pred_augmented))


def parse_paper_overrides(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--paper expects paper_id=path, got: {item!r}")
        pid, p = item.split("=", 1)
        out[pid] = Path(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paper",
        action="append",
        default=[],
        help="Override paper_id=path. Example: --paper bert=papers/BERT.pdf",
    )
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument(
        "--delay",
        type=float,
        default=10.0,
        help="Seconds to wait between Gemini calls (avoid rate limits).",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the no-profile baseline run (e.g. when you only want to inspect the augmented behavior).",
    )
    args = ap.parse_args()

    config.validate_required("GEMINI_API_KEY")

    examples = load_dataset(DATASET_PATH)
    print(f"Loaded {len(examples)} examples from {DATASET_PATH}")

    overrides = parse_paper_overrides(args.paper)
    paper_paths = {**DEFAULT_PAPER_PATHS, **overrides}

    profiles: dict[str, ContributionProfile] = {}
    titles_abstracts: dict[str, tuple[str, str]] = {}
    for ex in examples:
        if ex.paper_id in profiles:
            continue
        if ex.paper_id not in paper_paths:
            raise SystemExit(
                f"No PDF path configured for paper_id={ex.paper_id!r}. "
                f"Pass --paper {ex.paper_id}=<path>."
            )
        profile = load_or_build_profile(ex.paper_id, paper_paths[ex.paper_id])
        profiles[ex.paper_id] = profile

        # Reuse parsed paper for title/abstract if cache miss; otherwise re-parse to get them.
        # Cheap path: title/abstract are also recoverable from PDF; cache them too.
        ta_cache = PROFILES_DIR / f"{ex.paper_id}.title_abstract.json"
        if ta_cache.exists():
            ta = json.loads(ta_cache.read_text(encoding="utf-8"))
            titles_abstracts[ex.paper_id] = (ta["title"], ta["abstract"])
        else:
            parsed = GrobidPDFParser(str(paper_paths[ex.paper_id])).parse()
            titles_abstracts[ex.paper_id] = (parsed.title, parsed.abstract)
            ta_cache.write_text(
                json.dumps({"title": parsed.title, "abstract": parsed.abstract}, indent=2),
                encoding="utf-8",
            )

        print(f"\n[profile/{ex.paper_id}]")
        print(f"  proposes: {list(profile.proposes)}")
        print(f"  uses:     {list(profile.uses)}")

    by_paper: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_paper[ex.paper_id].append(ex)

    pred_baseline_all: list[str] = []
    pred_augmented_all: list[str] = []
    examples_in_order: list[Example] = []

    for paper_id, paper_examples in by_paper.items():
        sentences = [to_sentence_record(ex) for ex in paper_examples]
        title, abstract = titles_abstracts[paper_id]

        print(f"\n=== Classifying {len(sentences)} sentences for paper_id={paper_id!r} ===")

        if args.skip_baseline:
            preds_base = ["SKIPPED"] * len(sentences)
        else:
            print("--- baseline (no contribution profile) ---")
            preds_base = classify_with_prompt(
                sentences, title, abstract,
                system_prompt=None,
                batch_size=args.batch_size,
                delay=args.delay,
            )

        print("--- augmented (with contribution profile) ---")
        augmented_prompt = build_classifier_system_prompt(profiles[paper_id])
        preds_aug = classify_with_prompt(
            sentences, title, abstract,
            system_prompt=augmented_prompt,
            batch_size=args.batch_size,
            delay=args.delay,
        )

        examples_in_order.extend(paper_examples)
        pred_baseline_all.extend(preds_base)
        pred_augmented_all.extend(preds_aug)

    report(examples_in_order, pred_baseline_all, pred_augmented_all)


if __name__ == "__main__":
    main()
