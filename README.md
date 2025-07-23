
# CellNiche

## Overview
**CellNiche** is a scalable, **cell-centric** framework for identifying and characterizing cellular micro-environments from **atlas-scale, heterogeneous spatial omics data**.  
Instead of processing entire tissue slices, CellNiche samples **local subgraphs** around each cell and learns **context-aware embeddings** via **contrastive learning**, while explicitly **decoupling molecular identity** (gene expression or cell-type labels) from **spatial proximity modeling**.

### Key Features


## Installation
### From Source
```bash
git clone https://github.com/Super-LzzZ/CellNiche.git
cd cellniche
```
### From PyPI
```bash
pip install CellNiche
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
```bash
python ./cellniche/main.py --config ./configs/xxx.yaml

```

```python
import CellNiche

# Parse arguments from a YAML config
opts = cellniche.parse_args([
    "--config", "configs/xxx.yaml"
])
# Run training/inference
cellniche.main(opts)

```

Example YAML snippet (configs/example.yaml):
```yaml
# Data & preprocessing
data_path: "path/data/"
dataset: "osmFISH_SScortex"
phenoLabels: "ClusterName"
nicheLabels: "Region"
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
seed: 42
save: False
metrics: true
refine: False
save_path: "path/results"
verbose: true
```
