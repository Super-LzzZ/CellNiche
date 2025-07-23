import os
import random
import numpy as np
import pandas as pd
import torch
import scipy
import logging
import torch.nn.functional as F
import scanpy as sc
from natsort import natsorted
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    f1_score,
    silhouette_score,
)
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected

from typing import Optional, Union, Any, Tuple, List

def load_data(
    data_path: str,
    dataset: str,
    phenoLabels: str,
    nicheLabels: Optional[str],
    embedding_type: str,
    radius: Optional[float],
    k_neighborhood: int,
    hvg: bool
) -> Tuple[torch.FloatTensor, torch.LongTensor, np.ndarray, Any, int, Optional[torch.FloatTensor]]:
    """
    Load AnnData from .h5ad, build node features, edge index, and labels.

    Args:
        data_path: Directory containing the .h5ad files.
        dataset: Filename (without extension) to load.
        phenoLabels: Column name in `adata.obs` for phenotype labels (one-hot).
        nicheLabels: Column name in `adata.obs` for true labels (or None to skip).
        embedding_type: One of 'pheno_expr', 'pheno', 'expr'.
        radius: Radius threshold (if not None) for radius graph.
        k_neighborhood: Number of neighbors for kNN graph if radius is None.
        hvg: Whether to select highly variable genes for expression.

    Returns:
        x: Node feature matrix (one-hot or expr) as FloatTensor.
        edge_index: LongTensor[2, E] of graph edges.
        y: True label array of shape [N].
        adata: The AnnData object.
        n_classes: Number of unique labels in y.
        expr: Expression matrix as FloatTensor if needed, else None.
    """
    # 1) Read AnnData
    path = os.path.join(data_path, f"{dataset}.h5ad")
    adata = sc.read_h5ad(path).copy()

    # 2) Build phenotype one-hot features
    pheno = adata.obs[phenoLabels].astype(str)
    ph_le = LabelEncoder().fit(pheno)
    ph_idx = ph_le.transform(pheno)
    onehot = torch.nn.functional.one_hot(
        torch.from_numpy(ph_idx), num_classes=len(ph_le.classes_)
    ).float()

    # 3) Build graph edge_index
    if "edgeList" in adata.uns:
        edge_np = np.array(adata.uns["edgeList"])
        edge_index = torch.from_numpy(edge_np).long()
        edge_index = to_undirected(edge_index)
    else:
        # choose coords
        if "spatial" in adata.obsm and adata.obsm["spatial"] is not None:
            coords = adata.obsm["spatial"]
            # coords = adata.obs[["x", "y"]].to_numpy() # spleen
        else:
            coords = adata.obs[["x", "y"]].to_numpy()

        if radius is not None:
            nbrs = NearestNeighbors(radius=radius).fit(coords)
            _, idxs = nbrs.radius_neighbors(coords)
            rows = np.concatenate([np.full(len(n), i) for i, n in enumerate(idxs)])
            cols = np.concatenate(idxs)
        else:
            nbrs = NearestNeighbors(n_neighbors=k_neighborhood+1).fit(coords)
            _, idxs = nbrs.kneighbors(coords)
            rows = np.repeat(np.arange(coords.shape[0]), k_neighborhood)
            cols = idxs[:, 1:].flatten()

        mat = coo_matrix((np.ones_like(rows), (rows, cols)),
                         shape=(coords.shape[0], coords.shape[0]))
        mat = mat + mat.T  # make undirected
        edge_index = torch.from_numpy(np.vstack(mat.nonzero()).astype(np.int64))
        
        neighbors_count = np.array([len(neighbors) for neighbors in idxs])
        average_neighbors = neighbors_count.mean()
        logging.info(f"Average number of neighbors per node: {average_neighbors}")
        # print(f"================ Average number of neighbors per node: {average_neighbors} ================")

    # 4) Encode true labels from nicheLabels if provided
    if nicheLabels is not None and nicheLabels in adata.obs:
        true_vals = adata.obs[nicheLabels].astype(str)
        nl_encoder = LabelEncoder().fit(true_vals)
        y = nl_encoder.transform(true_vals)
        n_classes = len(nl_encoder.classes_)
    else:
        # default dummy labels: all-zero, one class
        y = np.zeros(adata.n_obs, dtype=int)
        n_classes = 1

    # 5) Prepare expression matrix if needed
    expr: Optional[torch.FloatTensor] = None
    if embedding_type in ("pheno_expr", "expr"):
        if hvg:
            # select HVGs, normalize & log
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=512)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata = adata[:, adata.var["highly_variable"]]

        mat = adata.X
        if scipy.sparse.isspmatrix(mat):
            arr = mat.toarray()
        else:
            arr = np.array(mat)
        expr = torch.from_numpy(arr).float()

    # logging.info(
    #     f"Loaded {dataset}: " f"{onehot.shape[0]} nodes, {edge_index.shape[1]} edges, {onehot.shape[0].shape[-1]} features"
    # )
    
    # 6) Return according to embedding type
    if embedding_type == "pheno_expr":
        logging.info(
            f"Loaded {dataset}: " f"{onehot.shape[0]} nodes, {edge_index.shape[1]} edges, {onehot.shape[-1]} features"
        )
        return torch.tensor(onehot, dtype=torch.float), edge_index, y, adata, n_classes, torch.tensor(expr, dtype=torch.float)
    elif embedding_type == "pheno":
        logging.info(
            f"Loaded {dataset}: " f"{onehot.shape[0]} nodes, {edge_index.shape[1]} edges, {onehot.shape[-1]} features"
        )
        return torch.tensor(onehot, dtype=torch.float), edge_index, y, adata, n_classes, None
    elif embedding_type == "expr":
        logging.info(
            f"Loaded {dataset}: " f"{expr.shape[0]} nodes, {edge_index.shape[1]} edges, {expr.shape[-1]} features"
        )
        return torch.tensor(expr, dtype=torch.float), edge_index, y, adata, n_classes, None
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

def setup_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, Torch, and CUDA.

    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def create_sparse_tensor_from_edges(
    rows: list[int],
    cols: list[int],
    sparse_size: tuple[int, int],
    device: torch.device = torch.device("cpu"),
) -> SparseTensor:
    """
    Create a SparseTensor from row/col indices.

    Args:
        rows (list[int]): Row indices.
        cols (list[int]): Column indices.
        sparse_size (tuple[int,int]): Matrix size.
        device (torch.device): Device for tensor.

    Returns:
        SparseTensor
    """
    vals = torch.ones(len(rows), device=device)
    return SparseTensor(
        row=torch.tensor(rows, device=device),
        col=torch.tensor(cols, device=device),
        value=vals,
        sparse_sizes=sparse_size,
    )

def sparse_intersection_and_union(
    adj1: SparseTensor,
    adj2: SparseTensor,
    strategy: str = "and",
) -> SparseTensor:
    """
    Compute intersection or union of two sparse adjacency matrices.

    Args:
        adj1, adj2 (SparseTensor): Input graphs.
        strategy (str): 'and' or 'or'.

    Returns:
        SparseTensor
    """
    rows1, cols1 = adj1.storage.row(), adj1.storage.col()
    rows2, cols2 = adj2.storage.row(), adj2.storage.col()
    device = rows1.device

    set1 = set(zip(rows1.tolist(), cols1.tolist()))
    set2 = set(zip(rows2.tolist(), cols2.tolist()))

    if strategy == "and":
        common = set1 & set2
    else:
        common = set1 | set2

    if not common:
        return SparseTensor(sparse_sizes=adj1.sparse_sizes())

    rows, cols = zip(*common)
    return create_sparse_tensor_from_edges(rows, cols, adj1.sparse_sizes(), device)

def get_positivePairs(
    subAdj: SparseTensor,
    features: Optional[torch.Tensor] = None,
    strategy: str = "freq",
) -> SparseTensor:
    """
    Generate positive pair adjacency based on strategy.

    Args:
        subAdj (SparseTensor): Stochastic subgraph adjacency.
        features (Tensor): Node features.
        strategy (str): 'freq', 'sim', 'and', 'or'.

    Returns:
        SparseTensor: positive adjacency.
    """
    row, col, val = subAdj.storage.row(), subAdj.storage.col(), subAdj.storage.value()
    # Frequency-based mask
    freq_thresh = subAdj.sum(dim=1) / subAdj.storage.colptr()[1:]
    mask = val > freq_thresh[row]
    rows, cols, vals = row[mask], col[mask], val[mask]
    freq_adj = SparseTensor(row=rows, col=cols, value=vals, sparse_sizes=subAdj.sparse_sizes())

    if strategy == "freq":
        return freq_adj

    if features is None:
        raise ValueError("Features required for non-freq strategy.")

    # Similarity-based mask
    f_row, f_col = features[row], features[col]
    sim_vals = F.cosine_similarity(f_row, f_col, dim=1)
    sim_vals[row == col] = 0
    sim_adj = SparseTensor(row=row, col=col, value=sim_vals, sparse_sizes=subAdj.sparse_sizes())
    sim_thresh = sim_adj.sum(dim=1) / sim_adj.storage.colptr()[1:]
    sim_mask = sim_vals > sim_thresh[row]
    sim_rows, sim_cols, sim_vals = row[sim_mask], col[sim_mask], sim_vals[sim_mask]
    sim_based = SparseTensor(row=sim_rows, col=sim_cols, value=sim_vals, sparse_sizes=subAdj.sparse_sizes())

    if strategy == "sim":
        return sim_based

    # AND / OR combination
    return sparse_intersection_and_union(freq_adj, sim_based, strategy)

def match_labels(true_labels, predicted_labels, n_classes):
    from scipy.optimize import linear_sum_assignment as linear_assignment

    cost_matrix = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))

    row_ind, col_ind = linear_assignment(-cost_matrix)

    new_labels = np.copy(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        new_labels[predicted_labels == j] = i
    return new_labels

def refine_spatial_domains(y_pred, coord, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coord)
    distances, indices = nbrs.kneighbors(coord)
    indices = indices[:, 1:]

    y_refined = pd.Series(index=y_pred.index, dtype='object')

    for i in range(y_pred.shape[0]):
        y_pred_count = y_pred[indices[i, :]].value_counts()

        if y_pred[i] in y_pred_count.index:
            if (y_pred_count.loc[y_pred[i]] < n_neighbors / 2) and (y_pred_count.max() > n_neighbors / 2):
                y_refined[i] = y_pred_count.idxmax()
            else:
                # y_refined[i] = y_pred[i] # waring
                y_refined.iloc[i] = y_pred[i]
        else:
            y_refined.iloc[i] = y_pred[i]

    y_refined = pd.Categorical(
        values=y_refined.astype('U'),
        categories=natsorted(map(str, y_refined.unique())),
    )
    return y_refined

def clustering_st(
    adata: Any,
    n_clusters: int,
    features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    true_labels: Optional[np.ndarray] = None,
    refine: bool = False,
) -> Tuple[Any, dict]:
    """
    Perform KMeans clustering and compute evaluation metrics.

    Args:
        adata (AnnData): Annotated data object.
        features (Tensor or ndarray): Embeddings.
        n_clusters (int): Number of clusters.
        true_labels (ndarray): Ground-truth labels.

    Returns:
        adata (AnnData): Updated with 'kmeans' clusters.
        metrics (dict): Cluster evaluation metrics.
    """
    # 1) Convert to numpy
    if torch.is_tensor(features):
        feats = features.cpu().numpy()
    else:
        feats = features
    # 2) Run KMeans
    km = KMeans(n_clusters=n_clusters, random_state=0)
    raw_labels = km.fit_predict(feats).astype(int)

    # 3) Store raw labels
    adata.obs['kmeans'] = pd.Categorical(raw_labels)
    clustering_results = {'kmeans': raw_labels}

    # 4) Optional spatial refinement
    if refine:
        # spatial coords must exist
        coords = adata.obsm.get('spatial')
        if coords is None:
            raise ValueError("adata.obsm['spatial'] needed for refinement")
        for method, labels in list(clustering_results.items()):
            refined = refine_spatial_domains(pd.Series(labels), coords)
            refined = refined.astype(int)
            col = f"{method}_refined"
            adata.obs[col] = pd.Categorical(refined)
            clustering_results[col] = refined

    # 5) Compute metrics
    metrics_results: dict = {}
    if true_labels is not None:
        for method, labels in clustering_results.items():
            # align predicted → true
            aligned = match_labels(true_labels, labels, n_clusters)

            acc = (aligned == true_labels).mean()
            nmi = normalized_mutual_info_score(true_labels, aligned)
            ari = adjusted_rand_score(true_labels, aligned)
            ami = adjusted_mutual_info_score(true_labels, aligned)
            f1m = f1_score(true_labels, aligned, average='macro')
            f1i = f1_score(true_labels, aligned, average='micro')
            sil = silhouette_score(feats, aligned)

            metrics_results[method] = {
                'Acc': acc,
                'NMI': nmi,
                'AMI': ami,
                'ARI': ari,
                'F1 Macro': f1m,
                'F1 Micro': f1i,
                'Silhouette': sil,
            }

    return adata, metrics_results    
    
