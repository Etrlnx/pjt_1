import os
import sys
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
# Ensure graph-transformer modules are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
if GT_DIR not in sys.path:
    sys.path.append(GT_DIR)
@dataclass
class AnomalyScorerConfig:
    # Fusion weights (sum to 1.0)
    alpha_recon: float = 0.50     # Weight for reconstruction error
    beta_temporal: float = 0.25   # Weight for temporal deviation
    gamma_struct: float = 0.25    # Weight for graph structural deviation
    # Node feature importance weights matching the GAE loss
    feature_weights: tuple = (1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5)
    # Structural edge weight multiplier
    edge_loss_weight: float = 0.50
    # Minimum probability threshold for a transition to be considered "normal"
    rare_transition_prob_threshold: float = 1e-4

class TransitionBaseline:
    """
    Builds and maintains empirical transition matrices P(ID_j | ID_i)
    from benign ambient CAN graphs to detect abnormal message sequencing.
    """
    def __init__(self, vocab_size: int = 106):
        self.vocab_size = vocab_size
        self.transition_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        self.transition_probs = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        self.fitted = False
    def fit(self, benign_graphs: list):
        """Learns normal message transition probabilities from benign graphs."""
        for g in benign_graphs:
            if not hasattr(g, "edge_index") or g.edge_index.shape[1] == 0:
                continue
            src_nodes = g.edge_index[0].cpu().numpy()
            dst_nodes = g.edge_index[1].cpu().numpy()
            weights = g.edge_attr.cpu().numpy() if hasattr(g, "edge_attr") and g.edge_attr is not None else np.ones(len(src_nodes))
            # Map local node indices to global ID vocab indices
            src_ids = g.id_idx[src_nodes].cpu().numpy()
            dst_ids = g.id_idx[dst_nodes].cpu().numpy()
            for s, d, w in zip(src_ids, dst_ids, weights):
                if s < self.vocab_size and d < self.vocab_size:
                    self.transition_counts[s, d] += float(w)
                            # Normalize rows with Laplace smoothing
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        smoothed_counts = self.transition_counts + 1e-3
        smoothed_sums = row_sums + (1e-3 * self.vocab_size)
        self.transition_probs = smoothed_counts / np.maximum(smoothed_sums, 1e-6)
        self.fitted = True
    def compute_structural_penalty(self, g) -> float:
        """
        Computes the negative log-likelihood penalty for observed edges in graph g.
        High penalty indicates unexpected / unseen CAN ID sequences (masquerade signature).
        """
        if not self.fitted or g.edge_index.shape[1] == 0:
            return 0.0
        src_nodes = g.edge_index[0].cpu().numpy()
        dst_nodes = g.edge_index[1].cpu().numpy()
        weights = g.edge_attr.cpu().numpy() if hasattr(g, "edge_attr") and g.edge_attr is not None else np.ones(len(src_nodes))
        
        
        src_ids = g.id_idx[src_nodes].cpu().numpy()
        dst_ids = g.id_idx[dst_nodes].cpu().numpy()
        penalties = []
        for s, d, w in zip(src_ids, dst_ids, weights):
            if s < self.vocab_size and d < self.vocab_size:
                prob = self.transition_probs[s, d]
                # Negative log-probability weighted by transition frequency
                nll = -np.log(np.maximum(prob, 1e-7))
                penalties.append(nll * float(w))
        if not penalties:
            return 0.0
        total_weight = np.maximum(np.sum(weights), 1.0)
        # Normalized average transition NLL
        return float(np.sum(penalties) / total_weight)

class AnomalyScorer:
    """
    Evaluates individual graph windows and computes multi-component anomaly scores.
    """
    def __init__(self, config: AnomalyScorerConfig = None):
        self.config = config or AnomalyScorerConfig()
        self.feat_weights_tensor = torch.tensor(
            self.config.feature_weights, dtype=torch.float32
        )
    def compute_reconstruction_deviation(self, batch, outputs) -> tuple[float, torch.Tensor]:
        """
        Computes weighted node feature MSE and structural edge BCE error.
        Returns (scalar_reconstruction_error, per_node_residuals).
        """
        device = batch.x.device
        weights = self.feat_weights_tensor.to(device)
        # Per-node weighted squared residual
        node_residuals = (((outputs["x_recon"] - batch.x) ** 2) * weights).mean(dim=-1)
        node_error = node_residuals.mean().item()
        # Structural edge BCE loss if edge logits are available
        edge_error = 0.0
        if "edge_logits" in outputs and outputs["edge_logits"] is not None and batch.edge_index.shape[1] > 0:
            pos_labels = torch.ones_like(outputs["edge_logits"])
            edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], pos_labels)
            edge_error = edge_loss.item()
        total_recon_error = node_error + (self.config.edge_loss_weight * edge_error)
        return total_recon_error, node_residuals.detach()
    def compute_temporal_deviation(self, batch) -> float:
        """
        Evaluates inter-arrival time (IAT) variance and transmission irregularities.
        Feature indices in graph_builder:
          index 1: mean_iat
          index 2: std_iat
          index 7: activity_share
        """
        
        # std_iat (jitter) and activity share dispersion
        std_iat = batch.x[:, 2].abs().mean().item()
        activity = batch.x[:, 7].abs()
        # Measure entropy / concentration of activity share
        act_entropy = -(activity * torch.log(activity + 1e-6)).sum().item()
        # High jitter and sudden concentrated activity increase temporal deviation
        temporal_score = float(std_iat + 0.1 * act_entropy)
        return max(0.0, temporal_score)
    def score_window(self, batch, outputs, transition_baseline: TransitionBaseline = None) -> dict:
        """
        Computes all component deviations and produces the fused anomaly score.
        """
        r_error, node_residuals = self.compute_reconstruction_deviation(batch, outputs)
        t_error = self.compute_temporal_deviation(batch)
        g_error = transition_baseline.compute_structural_penalty(batch) if transition_baseline else 0.0
        fused_score = (
            self.config.alpha_recon * r_error +
            self.config.beta_temporal * t_error +
            self.config.gamma_struct * g_error
        )
        return {
            "anomaly_score": float(fused_score),
            "r_error": float(r_error),
            "t_error": float(t_error),
            "g_error": float(g_error),
            "node_residuals": node_residuals,
        }
