
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
- node2vec ≥ 0.5.0  
- tqdm ≥ 4.67.1  

You can install most dependencies with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install scanpy anndata scikit-learn numpy scipy pandas networkx node2vec tqdm
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
# Data settings
data_path: /path/to/data
dataset: my_sample
phenoLabels: cell_type
nicheLabels: region_annotation  # or null

# Embedding settings
embedding_type: pheno_expr
radius: 300
k_neighborhood: 20
hvg: true

# Model settings
hidden_channels: [512, 256]
projection: [128, 64]
decoder_hidden: [64]
tau: 0.9

# Training settings
epochs: 10
batchsize: 1024
lr: 0.001
wd: 0.0
max_steps: 100
max_duration: 60  # in minutes
```


## Requirements

## Requirements

## Requirements