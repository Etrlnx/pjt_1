"""
Training script — Graph Transformer + Graph Autoencoder (coupled)
--------------------------------------------------------------------
Training strategy, matching the project's core premise:

  - Train ONLY on benign windows. The autoencoder should learn what
    normal CAN bus behavior looks like; it never sees attack traffic
    during training.
  - Attack windows (including the masquerade captures — your zero-day
    stand-ins) are held out entirely and only used at evaluation time,
    to check whether reconstruction error on unseen attack traffic is
    meaningfully higher than on unseen benign traffic.

This script is intentionally a first pass / sanity-check level of
completeness: get a real signal that reconstruction error separates
benign from attack before investing in anomaly-score fusion (temporal +
structural deviation terms) or the explainability layer. That was the
agreed sequencing — don't build on top of this until the separation
here is actually meaningful.
"""

import pickle
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import os
from model import GraphTransformerAutoencoder, ModelConfig, reconstruction_loss
import torch.nn.functional as F

# Initial Configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GRAPH_DATA_PATH = os.path.join(OUTPUT_DIR, "graphs.pt")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.pkl")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
VAL_FRACTION = 0.15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def split_data(graphs):
    """
    Splits graphs strictly by capture.
    Only true ambient captures are partitioned into train, validation, and test_ambient.
    Attack captures are NEVER included in train or val splits (even during pre-attack windows),
    ensuring zero-leakage evaluation against unseen attack scenarios.
    """
    ambient_caps = sorted(list(set(g.capture_name for g in graphs if g.capture_name.startswith("ambient_"))))
    attack_caps = sorted(list(set(g.capture_name for g in graphs if not g.capture_name.startswith("ambient_"))))

    random.shuffle(ambient_caps)

    n_val = max(1, int(len(ambient_caps) * VAL_FRACTION))
    n_test = max(1, int(len(ambient_caps) * 0.15))

    val_captures = set(ambient_caps[:n_val])
    test_ambient_captures = set(ambient_caps[n_val:n_val + n_test])
    train_captures = set(ambient_caps[n_val + n_test:])

    train_benign = [g for g in graphs if g.capture_name in train_captures]
    val_benign = [g for g in graphs if g.capture_name in val_captures]
    # Test set contains held-out ambient drives + all attack captures
    test_graphs = [g for g in graphs if g.capture_name in test_ambient_captures or g.capture_name in attack_caps]

    print(f"Split captures: {len(train_captures)} train ambient, {len(val_captures)} val ambient, "
          f"{len(test_ambient_captures)} test ambient, {len(attack_caps)} attack captures.")
    print(f"Split graphs: {len(train_benign)} train (ambient), {len(val_benign)} val (ambient), "
          f"{len(test_graphs)} test total.")

    return train_benign, val_benign, test_graphs


def align_graph_feature_dim(graphs, expected_dim):
    """Backfills legacy graph files if their node-feature vector differs from
    the current model definition. This keeps old saved graph dumps usable while
    avoiding a runtime shape mismatch at the first forward pass."""
    aligned = 0
    for g in graphs:
        if g.x.size(-1) != expected_dim:
            if g.x.size(-1) < expected_dim:
                pad = torch.zeros((g.x.size(0), expected_dim - g.x.size(-1)), dtype=g.x.dtype)
                g.x = torch.cat([g.x, pad], dim=1)
            else:
                g.x = g.x[:, :expected_dim]
            aligned += 1
    if aligned:
        print(f"Aligned {aligned} graphs to feature dimension {expected_dim}.")


def normalize_graph_features(train_graphs, val_graphs, test_graphs):
    """
    Standardize node statistics using train-set statistics only.

    This is intentionally done before training begins and uses only the benign
    training captures, so the scale is anchored to the normal distribution seen
    during training instead of leaking validation/test/attack statistics into the
    feature transform.
    """
    if not train_graphs:
        raise ValueError("Training split is empty; cannot compute feature normalization stats.")

    train_x = torch.cat([g.x for g in train_graphs], dim=0)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    for graph_list in (train_graphs, val_graphs, test_graphs):
        for g in graph_list:
            g.x = ((g.x - mean) / std).to(torch.float32)

    return mean, std

# Training and evaluation

def run_epoch(model, loader, optimizer, config, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_graphs = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(DEVICE)

            outputs = model(
                x_stats=batch.x,
                id_idx=batch.id_idx,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
            )
            loss = reconstruction_loss(outputs, batch.x, batch.edge_index, config)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

    return total_loss / max(n_graphs, 1)


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Computes exact ROC-AUC using the Mann-Whitney U statistic in pure numpy."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = np.argsort(np.argsort(all_scores)) + 1.0
    pos_ranks = ranks[:len(pos_scores)]
    u = np.sum(pos_ranks) - len(pos_scores) * (len(pos_scores) + 1.0) / 2.0
    return float(u / (len(pos_scores) * len(neg_scores)))


def compute_per_graph_reconstruction_error(model, loader, config):
    """
    Computes per-graph reconstruction error and preserves the capture metadata
    for a per-capture evaluation breakdown. Use batch_size=1 during evaluation so
    plain string attributes like capture_name are retained as a single entry per
    graph rather than being silently collated away by PyG's default batching.
    """
    model.eval()
    results = []
    feature_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5],
        device=DEVICE,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if batch.num_graphs != 1:
                raise ValueError("Evaluation loader must use batch_size=1 to retain per-capture metadata.")

            outputs = model(
                x_stats=batch.x,
                id_idx=batch.id_idx,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
            )
            node_se = (((outputs["x_recon"] - batch.x) ** 2) * feature_weights).mean(dim=-1)
            node_error = node_se.mean().item()

            if config.reconstruct_edges and outputs["edge_logits"] is not None and batch.edge_index.shape[1] > 0:
                pos_labels = torch.ones_like(outputs["edge_logits"])
                edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], pos_labels).item()
                graph_error = node_error + 0.5 * edge_loss
            else:
                graph_error = node_error

            raw_label = batch.y[0].item()
            raw_capture = batch.capture_name
            if isinstance(raw_capture, (list, tuple)):
                capture_name = raw_capture[0]
            else:
                capture_name = raw_capture

            results.append((graph_error, raw_label, capture_name))

    return results


# Driver function

def main():
    set_seed(SEED)

    print("Loading graphs and vocab...")
    graphs = torch.load(GRAPH_DATA_PATH, weights_only=False)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs, vocab size {len(vocab)}")

    config = ModelConfig()
    config.num_ids = len(vocab)

    align_graph_feature_dim(graphs, config.node_stat_feature_dim)
    train_graphs, val_graphs, test_graphs = split_data(graphs)
    train_mean, train_std = normalize_graph_features(train_graphs, val_graphs, test_graphs)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    print(f"Train-set feature stats: mean={train_mean[:4].tolist()} | std={train_std[:4].tolist()}")

    model = GraphTransformerAutoencoder(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {DEVICE}...")
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, config, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, config, train=False)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
            }, CHECKPOINT_PATH)

    # ------------------------------------------------------------------
    # Sanity check: does reconstruction error separate benign from attack
    # on held-out data?
    # ------------------------------------------------------------------
    print("\nEvaluating on test set (held-out benign + all attack windows)...")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    graph_results = compute_per_graph_reconstruction_error(model, test_loader, config)
    errors = np.array([result[0] for result in graph_results], dtype=np.float32)
    labels = np.array([result[1] for result in graph_results], dtype=np.int64)
    capture_names = [result[2] for result in graph_results]

    per_capture_errors = {}
    for err, label, capture_name in graph_results:
        per_capture_errors.setdefault(capture_name, []).append((err, label))

    print("\nPer-capture reconstruction breakdown:")
    for capture_name in sorted(per_capture_errors):
        values = np.array([entry[0] for entry in per_capture_errors[capture_name]], dtype=np.float32)
        n_benign = sum(1 for entry in per_capture_errors[capture_name] if entry[1] == 0)
        n_attack = sum(1 for entry in per_capture_errors[capture_name] if entry[1] == 1)
        print(f"  {capture_name:45s}: mean={values.mean():.4f}, std={values.std():.4f}, "
              f"n_benign={n_benign}, n_attack={n_attack}")

    benign_errors = errors[labels == 0]
    attack_errors = errors[labels == 1]

    roc_auc = compute_roc_auc(errors, labels)

    print(f"\nAggregate Evaluation Summary:")
    print(f"  Benign reconstruction error : mean={benign_errors.mean():.4f}, std={benign_errors.std():.4f}, n={len(benign_errors)}")
    print(f"  Attack reconstruction error : mean={attack_errors.mean():.4f}, std={attack_errors.std():.4f}, n={len(attack_errors)}")
    print(f"  Reconstruction ROC-AUC      : {roc_auc:.4f}")

    if attack_errors.mean() > benign_errors.mean():
        print("\n-> Attack windows show HIGHER reconstruction error on average.")
    else:
        print("\n-> Attack windows do NOT show higher reconstruction error on average.")

if __name__ == "__main__":
    main()
    