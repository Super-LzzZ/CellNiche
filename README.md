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

A successful example
```bash
conda create -n cellniche python=3.9
conda activate cellniche
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install torch_cluster-1.6.3+pt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.18+pt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch-geometric==2.6.1
pip install CellNiche

pip install pyyaml
...
```


## Tutorials
Coming soon


## Getting Started
### bash(recommend)
```bash
python ./cellniche/main.py --config ./configs/xxx.yaml

```
### python
```python
import cellniche

# Parse arguments from a YAML config
# Run training/inference
cellniche.main(["--config", "./configs/xxx.yaml"])

```

