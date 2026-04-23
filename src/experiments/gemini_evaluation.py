"""Evaluate GeminiClassifier against labeled citation datasets (SciCite, ACL-ARC)."""

import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation import (
    load_scicite_jsonl,
    load_acl_arc_jsonl,
    ClassifierEvalExample,
)
from entities import SentenceRecord, ParsedPaper, CitationIntent
from experiments.gemini_classifier import GeminiClassifier


@dataclass
class ClassifierMetrics:
    """Metrics for classification evaluation."""
    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    macro_f1: float
    weighted_f1: float
    confusion_matrix: dict[str, dict[str, int]]
    total_examples: int
    correctly_classified: int


def compute_intent_metrics(
    gold_labels: list[CitationIntent | None],
    predictions: list[CitationIntent | None],
) -> ClassifierMetrics:
    """Compute classification metrics for citation intent prediction."""
    
    # Filter out None labels
    valid_pairs = [
        (gold, pred) for gold, pred in zip(gold_labels, predictions)
        if gold is not None and pred is not None
    ]
    
    if not valid_pairs:
        raise ValueError("No valid examples with gold labels")
    
    gold_filtered = [gold for gold, pred in valid_pairs]
    
    # Compute accuracy
    correct = sum(1 for g, p in valid_pairs if g == p)
    accuracy = correct / len(valid_pairs)
    
    # Get all intent classes
    all_intents = set(CitationIntent)
    
    # Compute per-class metrics
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    
    confusion_matrix: dict[str, dict[str, int]] = {}
    for intent in all_intents:
        confusion_matrix[intent.name] = {i.name: 0 for i in all_intents}
    
    for gold, pred in valid_pairs:
        confusion_matrix[gold.name][pred.name] += 1
    
    for intent in all_intents:
        intent_name = intent.name
        
        # TP: predicted this intent and was correct
        tp = confusion_matrix[intent_name][intent_name]
        
        # FP: predicted this intent but was wrong
        fp = sum(
            confusion_matrix[other_name][intent_name]
            for other_name in confusion_matrix
            if other_name != intent_name
        )
        
        # FN: should have predicted this intent but didn't
        fn = sum(
            confusion_matrix[intent_name][pred_name]
            for pred_name in confusion_matrix[intent_name]
            if pred_name != intent_name
        )
        
        # Precision
        if tp + fp > 0:
            precision[intent_name] = tp / (tp + fp)
        else:
            precision[intent_name] = 0.0
        
        # Recall
        if tp + fn > 0:
            recall[intent_name] = tp / (tp + fn)
        else:
            recall[intent_name] = 0.0
        
        # F1
        if precision[intent_name] + recall[intent_name] > 0:
            f1[intent_name] = 2 * (precision[intent_name] * recall[intent_name]) / (
                precision[intent_name] + recall[intent_name]
            )
        else:
            f1[intent_name] = 0.0
    
    # Macro F1 (unweighted average)
    macro_f1 = sum(f1.values()) / len(f1) if f1 else 0.0
    
    # Weighted F1
    class_counts: dict[str, int] = defaultdict(int)
    for gold in gold_filtered:
        class_counts[gold.name] += 1
    
    weighted_f1 = sum(
        f1[intent_name] * class_counts[intent_name]
        for intent_name in f1
    ) / len(valid_pairs) if valid_pairs else 0.0
    
    return ClassifierMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix={k: dict(v) for k, v in confusion_matrix.items()},
        total_examples=len(valid_pairs),
        correctly_classified=correct,
    )


def compute_worthiness_metrics(
    gold_worthy: list[bool | None],
    predictions_worthy: list[bool | None],
) -> dict[str, float]:
    """Compute binary classification metrics for citation worthiness."""
    
    # Filter out None labels
    valid_pairs = [
        (gold, pred) for gold, pred in zip(gold_worthy, predictions_worthy)
        if gold is not None
    ]
    
    if not valid_pairs:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    gold_filtered, pred_filtered = zip(*valid_pairs)
    
    # Accuracy
    correct = sum(1 for g, p in valid_pairs if g == p)
    accuracy = correct / len(valid_pairs)
    
    # Binary metrics: worthy = True (positive class)
    tp = sum(1 for g, p in valid_pairs if g and p)  # true positives
    fp = sum(1 for g, p in valid_pairs if not g and p)  # false positives
    fn = sum(1 for g, p in valid_pairs if g and not p)  # false negatives
    tn = sum(1 for g, p in valid_pairs if not g and not p)  # true negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def example_to_sentence_record(example: ClassifierEvalExample) -> SentenceRecord:
    """Convert a ClassifierEvalExample to SentenceRecord for classification."""
    return SentenceRecord(
        text=example.text,
        section=example.section or "UNKNOWN",
        position_in_section=0.5,  # placeholder
        has_citation=example.citation_worthy if example.citation_worthy is not None else True,
        citation_intent=example.citation_intent,
        citation_worthy=example.citation_worthy,
    )


def evaluate_classifier(
    examples: list[ClassifierEvalExample],
    classifier: GeminiClassifier,
    dummy_paper: ParsedPaper | None = None,
) -> tuple[ClassifierMetrics, dict[str, float]]:
    """
    Evaluate the classifier on a set of labeled examples.
    
    Args:
        examples: List of ClassifierEvalExample with ground-truth labels
        classifier: GeminiClassifier instance
        dummy_paper: A ParsedPaper to use for context (if None, creates minimal dummy)
    
    Returns:
        Tuple of (intent_metrics, worthiness_metrics)
    """
    
    if dummy_paper is None:
        # Create a minimal dummy paper for context
        dummy_paper = ParsedPaper(
            title="Evaluation Dataset",
            abstract="This is a dataset of labeled sentences for evaluation.",
            sections={},
            references=[],
        )
    
    # Convert examples to SentenceRecords
    sentences = [example_to_sentence_record(example) for example in examples]
    
    print(f"Classifying {len(sentences)} sentences...")
    # Classify in batches
    classified = classifier.classify_sentences(sentences, dummy_paper)
    
    # Extract predictions and gold labels
    gold_intents = [ex.citation_intent for ex in examples]
    predicted_intents = [s.citation_intent for s in classified]
    
    gold_worthy = [ex.citation_worthy for ex in examples]
    predicted_worthy = [s.citation_worthy for s in classified]
    
    # Compute metrics
    print("Computing metrics...")
    intent_metrics = compute_intent_metrics(gold_intents, predicted_intents)
    worthiness_metrics = compute_worthiness_metrics(gold_worthy, predicted_worthy)
    
    return intent_metrics, worthiness_metrics


def print_metrics(
    intent_metrics: ClassifierMetrics,
    worthiness_metrics: dict[str, float],
):
    """Pretty-print evaluation results."""
    print("\n" + "="*80)
    print("CITATION INTENT CLASSIFICATION RESULTS")
    print("="*80)
    print(f"Total Examples: {intent_metrics.total_examples}")
    print(f"Correctly Classified: {intent_metrics.correctly_classified}/{intent_metrics.total_examples}")
    print(f"Accuracy: {intent_metrics.accuracy:.4f}")
    print(f"Macro F1: {intent_metrics.macro_f1:.4f}")
    print(f"Weighted F1: {intent_metrics.weighted_f1:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Intent':<20} {'Precision':<15} {'Recall':<15} {'F1':<15}")
    print("-" * 80)
    for intent_name in sorted(intent_metrics.f1.keys()):
        p = intent_metrics.precision.get(intent_name, 0.0)
        r = intent_metrics.recall.get(intent_name, 0.0)
        f = intent_metrics.f1.get(intent_name, 0.0)
        print(f"{intent_name:<20} {p:<15.4f} {r:<15.4f} {f:<15.4f}")
    
    print("\nConfusion Matrix:")
    print("-" * 80)
    intents = sorted(intent_metrics.confusion_matrix.keys())
    header = "Gold \\ Pred" + "".join(f"{i:<12}" for i in intents)
    print(header)
    print("-" * 80)
    for gold in intents:
        row = f"{gold:<12}"
        for pred in intents:
            count = intent_metrics.confusion_matrix[gold].get(pred, 0)
            row += f"{count:<12}"
        print(row)
    
    print("\n" + "="*80)
    print("CITATION WORTHINESS CLASSIFICATION RESULTS")
    print("="*80)
    print(f"Accuracy: {worthiness_metrics['accuracy']:.4f}")
    print(f"Precision: {worthiness_metrics['precision']:.4f}")
    print(f"Recall: {worthiness_metrics['recall']:.4f}")
    print(f"F1 Score: {worthiness_metrics['f1']:.4f}")
    print(f"\nTP: {worthiness_metrics.get('tp', 0)}, FP: {worthiness_metrics.get('fp', 0)}")
    print(f"FN: {worthiness_metrics.get('fn', 0)}, TN: {worthiness_metrics.get('tn', 0)}")


def evaluate_on_scicite(
    dataset_path: str,
    batch_size: int = 10,
    max_examples: int = 100,
    delay_between_calls_seconds: float = 60.0,
):
    """Evaluate classifier on SciCite dataset."""
    print(f"Loading SciCite dataset from {dataset_path}...")
    examples = load_scicite_jsonl(dataset_path)
    
    if not examples:
        print("No examples loaded!")
        return
    
    # Limit to examples with both labels
    filtered_examples = [
        ex for ex in examples
        if ex.citation_intent is not None and ex.citation_worthy is not None
    ]

    print(f"Loaded {len(filtered_examples)} examples with complete labels")

    if max_examples is not None:
        filtered_examples = filtered_examples[:max_examples]
        print(f"Evaluating first {len(filtered_examples)} examples")

    classifier = GeminiClassifier(
        model="gemini-3.1-flash-lite-preview",
        batch_size=batch_size,
        delay_between_calls_seconds=delay_between_calls_seconds,
    )
    intent_metrics, worthiness_metrics = evaluate_classifier(filtered_examples, classifier)
    
    print_metrics(intent_metrics, worthiness_metrics)
    return intent_metrics, worthiness_metrics


def evaluate_on_acl_arc(
    dataset_path: str,
    batch_size: int = 10,
    max_examples: int = 100,
    delay_between_calls_seconds: float = 60.0,
):
    """Evaluate classifier on ACL-ARC dataset."""
    print(f"Loading ACL-ARC dataset from {dataset_path}...")
    examples = load_acl_arc_jsonl(dataset_path)
    
    if not examples:
        print("No examples loaded!")
        return
    
    # Limit to examples with both labels
    filtered_examples = [
        ex for ex in examples
        if ex.citation_intent is not None and ex.citation_worthy is not None
    ]

    print(f"Loaded {len(filtered_examples)} examples with complete labels")

    if max_examples is not None:
        filtered_examples = filtered_examples[:max_examples]
        print(f"Evaluating first {len(filtered_examples)} examples")

    classifier = GeminiClassifier(
        batch_size=batch_size,
        delay_between_calls_seconds=delay_between_calls_seconds,
    )
    intent_metrics, worthiness_metrics = evaluate_classifier(filtered_examples, classifier)
    
    print_metrics(intent_metrics, worthiness_metrics)
    return intent_metrics, worthiness_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate GeminiClassifier on labeled citation datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["scicite", "acl_arc"],
        default="scicite",
        help="Which dataset to evaluate on",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for classification",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of labeled examples to evaluate",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=60.0,
        help="Delay between Gemini API calls in seconds",
    )
    
    args = parser.parse_args()
    
    if not args.path:
        print("Error: --path is required")
        parser.print_help()
        sys.exit(1)
    
    if args.dataset == "scicite":
        evaluate_on_scicite(
            args.path,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            delay_between_calls_seconds=args.delay_seconds,
        )
    elif args.dataset == "acl_arc":
        evaluate_on_acl_arc(
            args.path,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            delay_between_calls_seconds=args.delay_seconds,
        )
