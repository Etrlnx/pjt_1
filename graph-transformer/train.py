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

# Initial Configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DATA_PATH = os.path.join(BASE_DIR,"outputs","graphs.pt")
VOCAB_PATH = os.path.join(BASE_DIR,"outputs","vocab.pkl")
CHECKPOINT_PATH = os.path.join(BASE_DIR,"outputs")

BATCH_SIZE = 16         
NUM_EPOCHS = 30         
LEARNING_RATE = 2e-4    
VAL_FRACTION = 0.15  
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Data splitting

def split_data(graphs):
    """
    Splits graphs into:
      - train: benign only
      - val:   held-out benign (for early stopping / loss monitoring)
      - test:  held-out benign + ALL attack windows (for the actual
               benign-vs-attack separation check)

    TODO: Once you move past this sanity-check stage, consider splitting
    by CAPTURE rather than by individual window for train/val — windows
    from the same capture (especially overlapping benign windows) are
    correlated, so a per-window random split likely overstates validation
    performance. Kept simple here since this stage's goal is just "does
    reconstruction error separate benign from attack at all."
    """
    
    benign = [g for g in graphs if g.y.item() == 0]
    attack = [g for g in graphs if g.y.item() == 1]

    random.shuffle(benign)
    n_val = int(len(benign) * VAL_FRACTION)

    val_benign = benign[:n_val]
    remaining_benign = benign[n_val:]

    # Further split remaining_benign into train / held-out-test-benign
    n_test_benign = int(len(remaining_benign) * 0.1)
    test_benign = remaining_benign[:n_test_benign]
    train_benign = remaining_benign[n_test_benign:]

    test_set = test_benign + attack

    print(f"Split: {len(train_benign)} train (benign), "
          f"{len(val_benign)} val (benign), "
          f"{len(test_set)} test ({len(test_benign)} benign + {len(attack)} attack)")

    return train_benign, val_benign, test_set

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


def compute_per_graph_reconstruction_error(model, loader):
    """
    Computes per-graph mean squared reconstruction error — this is the
    raw signal the anomaly score will eventually be built from.
    Returns a list of (error, label, capture_name) tuples.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(
                x_stats=batch.x,
                id_idx=batch.id_idx,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
            )
            # Per-node squared error, averaged per graph using PyG's batch vector
            node_se = ((outputs["x_recon"] - batch.x) ** 2).mean(dim=-1)  # [total_nodes_in_batch]

            for graph_id in batch.batch.unique():
                mask = batch.batch == graph_id
                graph_error = node_se[mask].mean().item()
                label = batch.y[graph_id].item()
                results.append(graph_error)
                # capture_name isn't batched by PyG automatically since
                # it's a plain string attribute, not a tensor — if you need
                # per-capture breakdown, iterate un-batched (batch_size=1)
                # instead. Left as-is here since this stage only needs the
                # benign-vs-attack error distributions, not per-capture detail.

    return results


# Driver function

def main():
    set_seed(SEED)

    print("Loading graphs and vocab...")
    graphs = torch.load(GRAPH_DATA_PATH)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs, vocab size {len(vocab)}")

    train_graphs, val_graphs, test_graphs = split_data(graphs)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    config = ModelConfig()
    config.num_ids = len(vocab)

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
    # on held-out data? This is the checkpoint before moving on to anomaly
    # score fusion or the explainability layer.
    # ------------------------------------------------------------------
    print("\nEvaluating on test set (held-out benign + all attack windows)...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])

    errors = compute_per_graph_reconstruction_error(model, test_loader)
    labels = [g.y.item() for g in test_graphs]

    errors = np.array(errors)
    labels = np.array(labels)

    benign_errors = errors[labels == 0]
    attack_errors = errors[labels == 1]

    print(f"\nBenign reconstruction error: mean={benign_errors.mean():.4f}, "
          f"std={benign_errors.std():.4f}, n={len(benign_errors)}")
    print(f"Attack reconstruction error: mean={attack_errors.mean():.4f}, "
          f"std={attack_errors.std():.4f}, n={len(attack_errors)}")

    if attack_errors.mean() > benign_errors.mean():
        print("\n-> Attack windows show HIGHER reconstruction error on average. "
              "This is the expected direction — worth checking the full "
              "distributions (not just means) and computing ROC-AUC next, "
              "rather than treating this print statement as a finished result.")
    else:
        print("\n-> Attack windows do NOT show higher reconstruction error. "
              "This means the model isn't yet capturing what makes attack "
              "traffic anomalous — worth revisiting node/edge feature design "
              "or training length before moving forward, not something to "
              "push past.")


if __name__ == "__main__":
    main()
