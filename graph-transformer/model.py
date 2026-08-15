"""
Graph Transformer + Graph Autoencoder — coupled model
--------------------------------------------------------------------
As discussed: the Graph Transformer (encoder) and Graph Autoencoder
(decoder) are trained jointly, not built/tuned in isolation, because
the encoder's weights are shaped by what the decoder needs to
reconstruct from.

Architecture:
    1. Node identity embedding (learned, indexed by global ID vocab)
       concatenated with the statistical node features from graph_builder.py.
    2. A stack of Graph Transformer layers (TransformerConv from
       PyTorch Geometric — multi-head attention restricted to graph edges).
    3. A decoder that reconstructs:
         a) the original node feature vector (main reconstruction signal)
         b) graph structure / edges (optional, off by default — see
            RECONSTRUCT_EDGES flag) via a dot-product edge decoder
            (standard Graph Autoencoder approach, Kipf & Welling 2016)

Only (a) is required to get a working anomaly score from reconstruction
error. (b) is included but flagged off by default — it's a reasonable
next experiment, not something to enable blindly, since it changes the
loss landscape and may need its own tuning pass.

Attention weights from the TransformerConv layers are retained on the
model instance after a forward pass — this is what the Explainability
Layer will consume later, so it's threaded through now rather than
retrofitted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

# Configuration


class ModelConfig:
    # Set this to len(vocab) from graph_builder.py's saved id_vocab.pkl
    num_ids = 106

    node_stat_feature_dim = 9

    # A slightly larger latent space helps the encoder distinguish abnormal
    # ID activity patterns without exploding the parameter count on CPU.
    id_embedding_dim = 20
    hidden_dim = 96
    latent_dim = 48
    num_transformer_layers = 4
    num_attention_heads = 4
    dropout = 0.08

    reconstruct_edges = True


# GRAPH TRANSFORMER
# Encoder

class GraphTransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        assert config.num_ids is not None, (
            "config.num_ids must be set to len(id_vocab) before constructing the model. "
            "See train.py for how this is wired up from graph_builder.py's saved vocab."
        )

        self.id_embedding = nn.Embedding(config.num_ids, config.id_embedding_dim)

        input_dim = config.node_stat_feature_dim + config.id_embedding_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        self.transformer_layers = nn.ModuleList()
        in_dim = config.hidden_dim
        for i in range(config.num_transformer_layers):
            out_dim = config.hidden_dim if i < config.num_transformer_layers - 1 else config.latent_dim
            # TransformerConv splits out_dim across heads internally when concat=True;
            # we request out_dim // heads per head so the final output is out_dim.
            self.transformer_layers.append(
                TransformerConv(
                    in_channels=in_dim,
                    out_channels=out_dim // config.num_attention_heads,
                    heads=config.num_attention_heads,
                    dropout=config.dropout,
                    edge_dim=1,        # we pass edge_weight as a 1-dim edge feature
                    concat=True,
                )
            )
            in_dim = out_dim

        self.dropout = nn.Dropout(config.dropout)

        # Populated after each forward() call — attention weights per layer.
        # Consumed later by the Explainability Layer.
        self.last_attention_weights = []

    def forward(self, x_stats, id_idx, edge_index, edge_weight):
        """
        x_stats:    [num_nodes, node_stat_feature_dim]
        id_idx:     [num_nodes]  (long tensor, indices into the ID vocab)
        edge_index: [2, num_edges]
        edge_weight:[num_edges]
        """
        id_emb = self.id_embedding(id_idx)                    # [num_nodes, id_embedding_dim]
        h = torch.cat([x_stats, id_emb], dim=-1)               # [num_nodes, node_stat_feature_dim + id_embedding_dim]
        h = F.relu(self.input_proj(h))

        edge_attr = edge_weight.unsqueeze(-1) if edge_weight.numel() > 0 else None

        self.last_attention_weights = []
        for i, layer in enumerate(self.transformer_layers):
            if edge_index.shape[1] == 0:
                # No edges in this graph (degenerate window) — TransformerConv
                # with zero edges still works but attention is vacuous; skip
                # attention capture for this case.
                h = layer(h, edge_index, edge_attr=None)
            else:
                h, (attn_edge_index, attn_weights) = layer(
                    h, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                self.last_attention_weights.append(
                    (attn_edge_index.detach(), attn_weights.detach())
                )
            if i < len(self.transformer_layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        return h  # [num_nodes, latent_dim] — the latent graph representation

# DECODER 
class GraphAutoencoderDecoder(nn.Module):
    """
    Reconstructs the original node statistical feature vector from the
    latent node embedding produced by the encoder.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.node_stat_feature_dim),
        )

    def forward(self, z):
        return self.mlp(z)  # [num_nodes, node_stat_feature_dim]


class EdgeDecoder(nn.Module):
    """
    Optional structural decoder (Kipf & Welling-style inner product decoder):
    reconstructs edge existence from node embeddings.
    Off by default — see ModelConfig.reconstruct_edges.
    """
    def forward(self, z, edge_index):
        src, dst = edge_index
        logits = (z[src] * z[dst]).sum(dim=-1)
        return logits  # raw logits, apply sigmoid outside if needed

# Model integration

class GraphTransformerAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GraphTransformerEncoder(config)
        self.decoder = GraphAutoencoderDecoder(config)
        self.edge_decoder = EdgeDecoder() if config.reconstruct_edges else None

    def forward(self, x_stats, id_idx, edge_index, edge_weight):
        z = self.encoder(x_stats, id_idx, edge_index, edge_weight)
        x_recon = self.decoder(z)

        edge_logits = None
        if self.edge_decoder is not None and edge_index.shape[1] > 0:
            edge_logits = self.edge_decoder(z, edge_index)

        return {
            "z": z,                    # latent node embeddings — reused by anomaly scoring + XAI later
            "x_recon": x_recon,        # reconstructed node features
            "edge_logits": edge_logits,
        }

    def get_attention_weights(self):
        """Convenience accessor for the Explainability Layer (built later)."""
        return self.encoder.last_attention_weights

# Loss metric

def reconstruction_loss(outputs: dict, x_stats: torch.Tensor, edge_index: torch.Tensor,
                         config: ModelConfig) -> torch.Tensor:
    """
    Node feature reconstruction loss (always active) + optional edge
    reconstruction loss (only if config.reconstruct_edges is True).

    We weight the anomaly-sensitive dimensions more heavily so the model does
    not overfit to generic volume statistics while underfitting the richer
    activity/attack cues that actually differ between benign and attack
    windows.
    """
    feature_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5],
        device=x_stats.device,
        dtype=x_stats.dtype,
    )
    node_residual = outputs["x_recon"] - x_stats
    node_loss = (node_residual.pow(2) * feature_weights).mean()

    if config.reconstruct_edges and outputs["edge_logits"] is not None:
        pos_labels = torch.ones_like(outputs["edge_logits"])
        edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], pos_labels)
        return node_loss + edge_loss

    return node_loss
