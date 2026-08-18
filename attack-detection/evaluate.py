"""
Comprehensive Zero-Day Attack Detection Benchmark
--------------------------------------------------------------------
Evaluates the full detection pipeline across held-out ambient drives and
unseen ROAD attack captures (including masquerade attacks).

Computes:
  - Binary Detection Metrics (Precision, Recall, F1, Accuracy)
  - Discrimination Metrics (ROC-AUC, PR-AUC)
  - Operational IDS Metrics (False Positive Rate, Detection Latency)
  - Per-Attack Family & Per-Capture Breakdown
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Ensure modules in graph-transformer and attack-detection are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
if GT_DIR not in sys.path:
    sys.path.append(GT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from model import GraphTransformerAutoencoder, ModelConfig
from train import split_data, normalize_graph_features, align_graph_feature_dim
from detector import ZeroDayDetector, RiskState, DetectionResult
from scorer import AnomalyScorerConfig


OUTPUTS_DIR = os.path.join(GT_DIR, "outputs")
GRAPH_DATA_PATH = os.path.join(OUTPUTS_DIR, "graphs.pt")
VOCAB_PATH = os.path.join(OUTPUTS_DIR, "vocab.pkl")
CHECKPOINT_PATH = os.path.join(OUTPUTS_DIR, "best_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC in pure NumPy for binary labels."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if scores.size == 0 or labels.size == 0:
        return 0.0
    if np.unique(labels).size < 2:
        return 0.0

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    pos_count = np.sum(sorted_labels == 1)
    neg_count = np.sum(sorted_labels == 0)
    if pos_count == 0 or neg_count == 0:
        return 0.0

    tp = 0.0
    fp = 0.0
    prev_score = None
    auc = 0.0

    for score, label in zip(sorted_scores[::-1], sorted_labels[::-1]):
        if prev_score is not None and score != prev_score:
            auc += (fp / neg_count) * (tp / pos_count)
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        prev_score = score

    auc += (fp / neg_count) * (tp / pos_count)
    return float(auc / (pos_count * neg_count))


def compute_pr_auc(scores: np.ndarray, labels: np.ndarray, num_thresholds: int = 200) -> float:
    """Computes exact Precision-Recall Area Under Curve in pure numpy."""
    pos_count = np.sum(labels == 1)
    if pos_count == 0:
        return 0.0

    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    precisions = []
    recalls = []

    for t in sorted(thresholds):
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precisions.append(p)
        recalls.append(r)

    # Sort by recall and integrate via trapezoidal rule
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    sorted_idx = np.argsort(recalls)
    return float(np.trapezoid(precisions[sorted_idx], recalls[sorted_idx]))


def run_evaluation():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("ZERO-DAY CAN INTRUSION DETECTION EVALUATION")
    print("=" * 70)

    if not os.path.exists(GRAPH_DATA_PATH) or not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Missing graph data or model checkpoint in {OUTPUTS_DIR}. "
            f"Please run graph_builder.py and train.py first."
        )

    print("\n[1/4] Loading graphs, vocabulary, and model weights...")
    graphs = torch.load(GRAPH_DATA_PATH, weights_only=False)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    config = ModelConfig()
    config.num_ids = len(vocab)
    align_graph_feature_dim(graphs, config.node_stat_feature_dim)

    train_graphs, val_graphs, test_graphs = split_data(graphs)
    train_mean, train_std = normalize_graph_features(train_graphs, val_graphs, test_graphs)

    model = GraphTransformerAutoencoder(config).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded {len(graphs)} total graphs. Vocab size: {len(vocab)}.")
    print(f"Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test.")

    print("\n[2/4] Initializing and Calibrating Zero-Day Detector...")
    scorer_config = AnomalyScorerConfig(
        alpha_recon=0.50,
        beta_temporal=0.25,
        gamma_struct=0.25,
    )
    detector = ZeroDayDetector(scorer_config)
    detector.fit_baseline(train_graphs, vocab_size=len(vocab))

    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    detector.calibrate(
        model, val_loader, device=DEVICE,
        suspicious_percentile=95.0,
        alert_percentile=99.0
    )

    print("\n[3/4] Evaluating Test Set (Held-Out Ambient Drives + 17 Attack Captures)...")
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    results: list[DetectionResult] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            outputs = model(
                x_stats=batch.x,
                id_idx=batch.id_idx,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
            )
            res = detector.evaluate_graph(batch, outputs)
            results.append(res)

    print("\n[4/4] Computing Intrusion Detection Metrics...")
    scores = np.array([r.anomaly_score for r in results], dtype=np.float64)
    labels = np.array([r.ground_truth_label for r in results], dtype=np.int64)

    # Predictions using calibrated alert threshold
    preds_binary = (scores >= detector.tau_suspicious).astype(int)

    tp = int(np.sum((preds_binary == 1) & (labels == 1)))
    fp = int(np.sum((preds_binary == 1) & (labels == 0)))
    tn = int(np.sum((preds_binary == 0) & (labels == 0)))
    fn = int(np.sum((preds_binary == 0) & (labels == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    tpr = recall

    roc_auc = compute_roc_auc(scores, labels)
    pr_auc = compute_pr_auc(scores, labels)

    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Total Test Windows      : {len(results)} (Benign: {np.sum(labels==0)}, Attack: {np.sum(labels==1)})")
    print(f"  True Positives (TP)     : {tp:5d}  |  False Positives (FP) : {fp:5d}")
    print(f"  False Negatives (FN)    : {fn:5d}  |  True Negatives (TN)  : {tn:5d}")
    print("-" * 70)
    print(f"  Precision               : {precision:.4f}")
    print(f"  Recall (Detection Rate) : {recall:.4f}")
    print(f"  F1-Score                : {f1:.4f}")
    print(f"  Accuracy                : {accuracy:.4f}")
    print(f"  False Positive Rate     : {fpr:.4f}")
    print("-" * 70)
    print(f"  ROC-AUC Discrimination  : {roc_auc:.4f}")
    print(f"  PR-AUC Score            : {pr_auc:.4f}")
    print("=" * 70)

    # Per-capture breakdown
    by_capture = {}
    for r in results:
        by_capture.setdefault(r.capture_name, []).append(r)

    print("\nPER-CAPTURE ZERO-DAY EVALUATION BREAKDOWN:")
    print(f"{'Capture Name':<45} | {'Type':<10} | {'Score Mean':<10} | {'Alerts/Total':<15} | {'Latency':<8}")
    print("-" * 95)

    for cap_name in sorted(by_capture):
        cap_results = by_capture[cap_name]
        cap_scores = [r.anomaly_score for r in cap_results]
        cap_labels = [r.ground_truth_label for r in cap_results]
        is_attack_cap = any(y == 1 for y in cap_labels)
        cap_type = "ATTACK" if is_attack_cap else "BENIGN"

        triggered_count = sum(1 for s in cap_scores if s >= detector.tau_suspicious)
        total_windows = len(cap_results)

        # Calculate time-to-detection latency (first trigger window relative to first attack window)
        latency_str = "N/A"
        if is_attack_cap:
            attack_start_times = [r.window_start for r in cap_results if r.ground_truth_label == 1]
            trigger_times = [r.window_start for r in cap_results if r.anomaly_score >= detector.tau_suspicious and r.ground_truth_label == 1]
            if attack_start_times and trigger_times:
                latency = max(0.0, trigger_times[0] - attack_start_times[0])
                latency_str = f"{latency:.2f}s"

        print(f"{cap_name:<45} | {cap_type:<10} | {np.mean(cap_scores):<10.4f} | {f'{triggered_count}/{total_windows}':<15} | {latency_str:<8}")

    print("=" * 95)
    print("\nEvaluation complete. Pipeline ready for Explainability Layer integration.")


if __name__ == "__main__":
    run_evaluation()
