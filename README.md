
# CellNiche

## Overview
CellNiche is a graph‐contrastive learning framework for spatial transcriptomics data. It constructs a cell‐cell graph from spatial coordinates or a provided adjacency, learns low-dimensional embeddings via a contrastive graph neural network (GNN), and can optionally reconstruct gene expression profiles. Use CellNiche to discover spatial domains and relationships between cells in high-resolution tissue maps.

## Installation
## From PyPI
```bash
pip install cellniche
```
## From Source
```bash
git clone https://github.com/Super-LzzZ/CellNiche.git
cd CellNiche/release
pip install .
```

## Requirements
- Python ≥ 3.7  
- PyTorch ≥ 1.12  
- PyTorch Geometric (torch-geometric, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv)  
- Scanpy ≥ 1.9  
- Anndata ≥ 0.9  
- scikit-learn ≥ 1.3  
- numpy ≥ 1.22  
- scipy ≥ 1.10  
- pandas ≥ 2.0  
- networkx ≥ 3.1   
- tqdm ≥ 4.67.1  

You can install most dependencies with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install scanpy anndata scikit-learn numpy scipy pandas networkx tqdm
```


## Getting Started
```python
import cellniche

# Parse arguments from a YAML config
opts = cellniche.parse_args([
    "--config", "configs/cortex.yaml"
])
# Run training/inference
cellniche.main(opts)

```

Example YAML snippet (configs/example.yaml):
```yaml
# Data & preprocessing
data_path: "path/data/cortex/"
dataset: "osmFISH_SScortex"
phenoLabels: "ClusterName"
nicheLabels: "Region" # None
embedding_type: "pheno_expr"
hvg: False

# Graph construction
k_neighborhood: null
radius: 1000.0

# Sampling & training
batch_size: 2048
epochs: null
max_steps: 20
lr: 0.001
weight_decay: 0.0
dropout: 0.0

# Model architecture
hidden_channels: [512, 256]
size: [10, 10]
projection: "" # [128, 64]
decoder: "" # [64]

# Contrastive strategy
tau: 0.9
negative_slope: 0.5
strategy: "freq"

use_weight: False
pos_weight_strategy: "inverse_sim"
neg_weight_strategy: "inverse_sim"

# Random‐walk
wt: 30
wl: 5
p: 0.25
q: 4.0

# Misc
seed: 3207
save: False
metrics: true
refine: False
save_path: "path/results/cortex"
verbose: true
```
