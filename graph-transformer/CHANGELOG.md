# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows its own milestone-based versioning during early
development (formal [Semantic Versioning](https://semver.org/) will apply
once a first stable pipeline exists).

## [Unreleased]

### Added
- Data preprocessing stage (`preprocess.py`): parses ROAD signal-translated
  CSV captures (ambient + attack), applies temporal windowing (overlapping for
  benign traffic, non-overlapping for attack traffic), and outputs windowed
  records ready for graph construction.
- Graph construction stage (`graph_builder.py`): converts windowed CAN traffic
  into PyTorch Geometric graphs. Nodes represent CAN arbitration IDs with a
  fixed-size statistical feature vector; edges are built from temporal message
  adjacency. Includes a global ID vocabulary for learned identity embeddings.
- Model definition (`model.py`): coupled Graph Transformer encoder (via
  `TransformerConv`) and Graph Autoencoder decoder, trained jointly against a
  node-feature reconstruction loss. Optional (disabled by default) structural
  edge-reconstruction term included for future experimentation.
- Training script (`train.py`): trains the model on benign traffic only, holds
  out benign + all attack windows (including ROAD's masquerade attacks) for
  evaluation, and reports whether reconstruction error separates benign from
  attack traffic as a first-pass sanity check.
- Project README documenting architecture, research motivation, research gap,
  and evaluation plan.

### Decided
- Selected the ROAD dataset (Oak Ridge National Laboratory) as the primary
  data source over the Car-Hacking dataset, due to its inclusion of masquerade
  attacks, which better match the project's zero-day / no-known-signature framing.
- Restricted the pipeline to ROAD's 17 signal-translated attack captures
  (plus all signal-translated ambient captures) rather than attempting to
  translate the remaining 16 raw-only captures, since doing so would require
  independently reproducing the CAN-D signal-extraction method rather than
  standard preprocessing.

### Changed
- Added train-split-only feature normalization in `train.py`: node statistics are
  standardized using the benign training set mean/std before training begins,
  preventing leakage from validation/test/attack graphs and preventing the model
  from being dominated by large-magnitude features.
- Updated reconstruction-error evaluation to preserve per-graph capture metadata by
  verifying a single-graph evaluation loader (`batch_size=1`) and recording
  `(error, label, capture_name)` entries for each graph.
- Added a per-capture summary in the held-out evaluation output so benign and
  attack captures can be inspected individually in addition to the aggregate
  benign-vs-attack mean comparison.

### Known limitations (tracked, not yet resolved)
- Graph construction schema (node feature definition, edge construction rule)
  is a first-pass design and has not yet been validated against training results.
- No formal evaluation (ROC-AUC, precision/recall, cross-dataset generalization)
  has been run yet — current status is a sanity check on reconstruction error
  separation only.

## [0.1.0] - Project scaffolding

### Added
- Initial README describing the overall architecture (dynamic graph
  construction, Graph Transformer, Graph Autoencoder, XAI evidence layer,
  gateway policy) and research motivation/gap.
- Initial architecture diagrams (Mermaid).
