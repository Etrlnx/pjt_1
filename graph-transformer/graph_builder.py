"""
Graph Construction
--------------------------------------------------------------------
Converts the WindowRecord objects from preprocess.py into
PyTorch Geometric `Data` graph objects, ready for the Graph Transformer.

Graph Architecture:

  Node = one CAN arbitration ID that appears at least once in the window.
  Node feature vector (fixed length, independent of how many decoded
  signal channels that ID has):
      [ msg_count,
        mean_inter_arrival_time, std_inter_arrival_time,
        signal_mean, signal_std, signal_min, signal_max,
        n_signal_channels ]
    Rationale: raw Signal_i_of_ID values can't be used directly as a
    fixed-size node feature because different IDs decode to different
    numbers of signals. Aggregating into generic statistics keeps the
    feature vector the same shape for every node regardless of ID.

  Node identity: each ID also gets an integer index into a global
  vocabulary (built once, across the whole dataset). This index is used
  by the model's embedding layer so the Graph Transformer can learn an
  ID-specific representation on top of the generic stats above.

  Edge = directed, built from temporal message adjacency: for every
  pair of consecutive messages (by timestamp) from two DIFFERENT IDs,
  add/increment a directed edge between them. Edge weight = count of
  such adjacent occurrences within the window, capturing how often one
  ID's messages are immediately followed by another's — a proxy for
  real CAN bus scheduling/arbitration patterns.

"""

import pickle
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from preprocess import WindowRecord

# Environment configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WINDOWED_DATA_PATH = os.path.join(OUTPUT_DIR, "road_windowed.pkl")
GRAPH_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "graphs.pt")
VOCAB_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vocab.pkl")
NODE_FEATURE_DIM = 9 # message count, inter-arrival stats, signal stats, activity share, signal range, signal channel count

# 1. ID Lookup table
def build_id_vocab(windows: list[WindowRecord]) -> dict:
    """
    Builds a global mapping from CAN arbitration ID (string)-to-integer index.
    Built once across ALL windows (ambient + attack) so the model's ID
    embedding table has a consistent, fixed vocabulary regardless of which
    window it's looking at.
    """
    all_ids = set()
    for idx, w in enumerate(windows):
        if w.messages is None:
            raise ValueError(
                f"Window {idx} ({w.capture_name}) has no raw messages. "
            ) 
        all_ids.update(w.messages["ID"].unique().tolist())

    vocab = {"<UNK>": 0}
    for i, id_str in enumerate(sorted(all_ids), start=1):
        vocab[id_str] = i

    return vocab

""" NOTE: If a genuinely novel/unseen ID appeared at deployment time (e.g.,
    an attacker spoofing an ID that never appears in this dataset), this
    fixed vocabulary would not have an embedding for it. That's a real
    limitation worth acknowledging — a reserved "unknown ID" index (index 0
    below) is included as a partial mitigation."""

# Feature Extraction

def _extract_signal_columns(df: pd.DataFrame) -> list[str]:
    """
    Extracts signal columns to process the ids of the messages within it
    """
    
    return [c for c in df.columns if c.startswith("Signal_")]


def extract_node_features(id_messages: pd.DataFrame, window_duration: float, total_window_messages: int) -> np.ndarray:
    """
    Computes a fixed-size statistical feature vector that is more sensitive to
    unusual ID behavior inside a window, not just raw volume.
    """
    signal_cols = _extract_signal_columns(id_messages)

    msg_count = len(id_messages)
    activity_share = float(msg_count / max(total_window_messages, 1))

    # Inter-arrival time stats
    times = id_messages["Time"].values
    if len(times) > 1:
        iats = np.diff(times)
        mean_iat = float(np.mean(iats))
        std_iat = float(np.std(iats))
    else:
        mean_iat = window_duration
        std_iat = 0.0

    if signal_cols:
        signal_values = id_messages[signal_cols].values.flatten()
        signal_values = signal_values[~np.isnan(signal_values)]
    else:
        signal_values = np.array([])

    if len(signal_values) > 0:
        signal_mean = float(np.mean(signal_values))
        signal_std = float(np.std(signal_values))
        signal_min = float(np.min(signal_values))
        signal_max = float(np.max(signal_values))
    else:
        signal_mean = signal_std = signal_min = signal_max = 0.0

    signal_range = float(signal_max - signal_min)
    n_signal_channels = float(len(signal_cols))

    return np.array([
        msg_count,
        mean_iat,
        std_iat,
        signal_mean,
        signal_std,
        signal_min,
        signal_max,
        activity_share,
        signal_range
    ], dtype=np.float32)


# 3. EDGE CONSTRUCTION

def build_edges(messages: pd.DataFrame, node_index: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds directed edges from temporal message adjacency.

    node_index: maps ID string -> local node index WITHIN THIS GRAPH
                (not the global vocabulary index — this is just 0..n_nodes-1
                for this specific window's graph).

    Returns:
        edge_index: [2, num_edges] array
        edge_weight: [num_edges] array (count of adjacent occurrences)
    """
    sorted_msgs = messages.sort_values("Time")
    ids_in_order = sorted_msgs["ID"].values

    edge_counts = {}
    for k in range(len(ids_in_order) - 1):
        src_id = ids_in_order[k]
        dst_id = ids_in_order[k + 1]
        if src_id == dst_id:
            continue  # skip self-loops from consecutive messages of the same ID
        src = node_index[src_id]
        dst = node_index[dst_id]
        edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1

    if not edge_counts:
        # Degenerate case: window had only one distinct ID, or every message
        # was from the same ID. Return an empty edge set — the model needs
        # to handle graphs with no edges gracefully.
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edges = np.array(list(edge_counts.keys())).T  # shape [2, num_edges]
    weights = np.array(list(edge_counts.values()), dtype=np.float32)

    return edges, weights


# ============================================================================
# 4. BUILD A SINGLE GRAPH FROM ONE WINDOW
# ============================================================================

def build_graph_for_window(window: WindowRecord, global_vocab: dict) -> Data:
    df = window.messages
    unique_ids = df["ID"].unique().tolist()

    # local node index for THIS graph (0..n_nodes-1)
    local_node_index = {id_str: i for i, id_str in enumerate(unique_ids)}

    window_duration = window.window_end - window.window_start
    total_window_messages = len(df)

    node_features = []
    global_id_indices = []  # for the model's ID embedding lookup

    for id_str in unique_ids:
        id_msgs = df[df["ID"] == id_str]
        feat = extract_node_features(id_msgs, window_duration, total_window_messages)
        node_features.append(feat)
        global_id_indices.append(global_vocab.get(id_str, global_vocab["<UNK>"]))

    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    id_idx = torch.tensor(global_id_indices, dtype=torch.long)

    edge_index_np, edge_weight_np = build_edges(df, local_node_index)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor([window.label], dtype=torch.long),
    )
    # Stash extra fields PyG doesn't know about natively — still accessible
    # as attributes on the Data object.
    data.id_idx = id_idx
    data.capture_name = window.capture_name
    data.window_start = window.window_start
    data.frac_attack_messages = window.frac_attack_messages
    data.graph_message_count = float(len(df))
    data.graph_unique_ids = float(len(unique_ids))
    data.graph_attack_fraction = float(df["Label"].mean())

    return data


# ============================================================================
# ============================================================================

# 5. Driver function

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(WINDOWED_DATA_PATH, "rb") as f:
        windows: list[WindowRecord] = pickle.load(f)

    print(f"Loaded {len(windows)} windows.")

    print("Building global ID vocabulary...")
    vocab = build_id_vocab(windows)
    print(f"  -> {len(vocab)} unique IDs (including <UNK>)")

    print("Building graphs...")
    graphs = []
    skipped = 0
    print("Processing windows in batches:\n")
    for i, w in enumerate(windows):
        try:
            g = build_graph_for_window(w, vocab)
            if g.x.shape[0] < 2: # node threshold to consider graph for computation
                skipped += 1
                continue
            graphs.append(g)
        except Exception as e:
            print(f"  WARNING: failed to build graph for window {i} "
                  f"({w.capture_name} @ {w.window_start}s): {e}")
            skipped += 1

        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(windows)}")

    print(f"Built {len(graphs)} graphs, skipped {skipped}.")

    torch.save(graphs, GRAPH_OUTPUT_PATH)
    with open(VOCAB_OUTPUT_PATH, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Saved graphs to {GRAPH_OUTPUT_PATH}")
    print(f"Saved vocab to {VOCAB_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
