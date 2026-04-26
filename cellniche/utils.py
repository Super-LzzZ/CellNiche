
import os
import random
import numpy as np
import pandas as pd
import torch
import scipy
import scipy.sparse as sps
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
from numpy.typing import ArrayLike
from scipy.stats import entropy
from sklearn.metrics import silhouette_samples


def to_float_tensor(arr):
    """Safely convert numpy array / tensor -> float32 tensor w/o grad."""
    if isinstance(arr, torch.Tensor):
        return arr.clone().detach().float()
    else:                       # numpy / list
        return torch.as_tensor(arr, dtype=torch.float)

def _resolve_squidpy_connectivity_key(adata, connectivity_key: str = "spatial") -> str:
    """
    Resolve Squidpy connectivity key in adata.obsp.

    Squidpy usually stores the spatial graph as:
        adata.obsp["spatial_connectivities"]

    If users pass connectivity_key="spatial", this function will look for
    "spatial_connectivities". If users directly pass "spatial_connectivities",
    it will also work.
    """
    if connectivity_key is None:
        connectivity_key = "spatial"

    # Case 1: user directly provides the obsp key
    if connectivity_key in adata.obsp:
        return connectivity_key

    # Case 2: user provides Squidpy key_added prefix, e.g. "spatial"
    candidate = f"{connectivity_key}_connectivities"
    if candidate in adata.obsp:
        return candidate

    raise KeyError(
        f"Cannot find Squidpy spatial connectivity graph in adata.obsp. "
        f"Tried '{connectivity_key}' and '{candidate}'. "
        f"Available obsp keys: {list(adata.obsp.keys())}. "
        f"For multi-slice data, please precompute the graph using "
        f"squidpy.gr.spatial_neighbors(..., library_key=..., key_added='{connectivity_key}')."
    )


def _edge_index_from_squidpy_obsp(
    adata,
    connectivity_key: str = "spatial",
) -> torch.LongTensor:
    """
    Build edge_index from a Squidpy-generated spatial connectivity matrix.

    Expected input:
        adata.obsp["spatial_connectivities"]
    or:
        adata.obsp[f"{connectivity_key}_connectivities"]

    Returns:
        edge_index: torch.LongTensor with shape [2, num_edges]
    """
    obsp_key = _resolve_squidpy_connectivity_key(adata, connectivity_key)

    adj = adata.obsp[obsp_key]

    # Ensure sparse matrix
    if not sps.issparse(adj):
        adj = sps.csr_matrix(adj)

    # Convert to boolean adjacency
    adj = adj.astype(bool)

    # Convert to LIL for efficient diagonal modification
    if isinstance(adj, sps.csr_matrix):
        adj = adj.tolil()
    else:
        adj = adj.tolil()

    # Remove self-loops
    adj.setdiag(0)

    # Back to CSR and remove explicit zeros
    adj = adj.tocsr()
    adj.eliminate_zeros()

    # Extract nonzero indices
    row, col = adj.nonzero()

    edge_index = torch.tensor(
        np.array([row, col]),
        dtype=torch.long,
    )

    logging.info(
        f"Using precomputed Squidpy graph from adata.obsp['{obsp_key}']: "
        f"{adata.n_obs} nodes, {edge_index.shape[1]} edges."
    )

    return edge_index

def load_data(
    data_path: str,
    dataset: str,
    phenoLabels: str,
    nicheLabels: Optional[str],
    embedding_type: str,
    radius: Optional[float],
    k_neighborhood: int,
    hvg: bool,
    n_hvg: int,
    multi_slice: bool = False,
    connectivity_key: str = "spatial",
    embedding_key: Optional[str] = None,
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
        multi_slice: bool = False,
        connectivity_key: str = "spatial",
        embedding_key: Optional[str] = None,

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
    valid_embedding_types = ["pheno_expr", "pheno", "expr", "embedding"]
    if embedding_type not in valid_embedding_types:
        raise ValueError(
            f"Unknown embedding_type: {embedding_type}. "
            f"Expected one of {valid_embedding_types}."
        )
    # 3) Build phenotype one-hot features only when needed
    onehot = None

    if embedding_type in ["pheno", "pheno_expr"]:
        if phenoLabels is None:
            raise ValueError(
                "phenoLabels is required when embedding_type is "
                f"'{embedding_type}', but got phenoLabels=None."
            )

        if phenoLabels not in adata.obs:
            raise KeyError(
                f"phenoLabels='{phenoLabels}' was not found in adata.obs. "
                f"Available obs columns: {list(adata.obs.columns)}"
            )

        pheno = adata.obs[phenoLabels].astype(str)
        ph_le = LabelEncoder().fit(pheno)
        ph_idx = ph_le.transform(pheno)

        onehot = torch.nn.functional.one_hot(
            torch.from_numpy(ph_idx),
            num_classes=len(ph_le.classes_)
        ).float()
        
        
    # 4) Prepare expression matrix if needed
    expr = None

    if embedding_type in ["pheno_expr", "expr"]:
        temp_adata = adata.copy()
        if hvg:
            sc.pp.highly_variable_genes(
                temp_adata,
                flavor="seurat_v3",
                n_top_genes=n_hvg,
            )
            sc.pp.normalize_total(temp_adata, target_sum=1e4)
            sc.pp.log1p(temp_adata)
            temp_adata = temp_adata[:, temp_adata.var["highly_variable"]].copy()

        mat = temp_adata.X

        if scipy.sparse.isspmatrix(mat):
            arr = mat.toarray()
        else:
            arr = np.asarray(mat)

        expr = torch.from_numpy(arr).float()
        del temp_adata
        
    # 5) For embedding_type="embedding", load features from adata.obsm[embedding_key]
    embedding = None

    if embedding_type == "embedding":
        if embedding_key is None:
            raise ValueError(
                "embedding_key is required when embedding_type='embedding'. "
                "Please provide the key of the embedding stored in adata.obsm, "
                "for example embedding_key='X_scVI' or embedding_key='X_pca'."
            )

        if embedding_key not in adata.obsm:
            raise KeyError(
                f"embedding_key='{embedding_key}' was not found in adata.obsm. "
                f"Available obsm keys: {list(adata.obsm.keys())}"
            )

        emb_arr = adata.obsm[embedding_key]

        if scipy.sparse.isspmatrix(emb_arr):
            emb_arr = emb_arr.toarray()
        else:
            emb_arr = np.asarray(emb_arr)

        if emb_arr.ndim != 2:
            raise ValueError(
                f"adata.obsm['{embedding_key}'] must be a 2D matrix, "
                f"but got shape {emb_arr.shape}."
            )

        if emb_arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm['{embedding_key}'] has {emb_arr.shape[0]} rows, "
                f"but adata has {adata.n_obs} cells."
            )

        embedding = torch.from_numpy(emb_arr).float()
    
        
    # 6) Build graph edge_index
    if multi_slice:
        logging.warning(
            "multi_slice=True detected. "
            "For multi-slice or multi-sample data, CellNiche expects a precomputed "
            "sample-aware spatial graph generated by Squidpy"
        )

        edge_index = _edge_index_from_squidpy_obsp(
            adata,
            connectivity_key=connectivity_key,
        )

    else:
        # Original single-slice logic
        if "edgeList" in adata.uns:
            edge_np = np.array(adata.uns["edgeList"])

            # Support both shape [2, E] and [E, 2]
            if edge_np.ndim != 2:
                raise ValueError("adata.uns['edgeList'] must be a 2D array.")

            if edge_np.shape[0] == 2:
                edge_index = torch.from_numpy(edge_np).long()
            elif edge_np.shape[1] == 2:
                edge_index = torch.from_numpy(edge_np.T).long()
            else:
                raise ValueError(
                    "adata.uns['edgeList'] should have shape [2, E] or [E, 2]."
                )

            edge_index = to_undirected(edge_index)

        else:
            # choose coords
            if "spatial" in adata.obsm and adata.obsm["spatial"] is not None:
                coords = adata.obsm["spatial"]
            else:
                coords = adata.obs[["x", "y"]].to_numpy()

            if radius is not None:
                nbrs = NearestNeighbors(radius=radius).fit(coords)
                _, idxs = nbrs.radius_neighbors(coords)

                rows_list, cols_list = [], []
                for i, neighbors in enumerate(idxs):
                    # remove self-loop
                    neighbors = neighbors[neighbors != i]
                    rows_list.extend([i] * len(neighbors))
                    cols_list.extend(neighbors.tolist())

                rows = np.asarray(rows_list)
                cols = np.asarray(cols_list)

            else:
                nbrs = NearestNeighbors(n_neighbors=k_neighborhood + 1).fit(coords)
                _, idxs = nbrs.kneighbors(coords)

                rows = np.repeat(np.arange(coords.shape[0]), k_neighborhood)
                cols = idxs[:, 1:].flatten()

            mat = coo_matrix(
                (np.ones_like(rows), (rows, cols)),
                shape=(coords.shape[0], coords.shape[0]),
            )
            mat = mat + mat.T  # make undirected
            edge_index = torch.from_numpy(
                np.vstack(mat.nonzero()).astype(np.int64)
            )

            neighbors_count = np.array([len(neighbors) for neighbors in idxs])
            average_neighbors = neighbors_count.mean()
            logging.info(f"Average number of neighbors per node: {average_neighbors}")

    # 7) Encode true labels from nicheLabels if provided
    if nicheLabels is not None and nicheLabels in adata.obs:
        true_vals = adata.obs[nicheLabels].astype(str)
        nl_encoder = LabelEncoder().fit(true_vals)
        y = nl_encoder.transform(true_vals)
        n_classes = len(nl_encoder.classes_)
    else:
        # default dummy labels: all-zero, one class
        y = np.zeros(adata.n_obs, dtype=int)
        n_classes = 1

    
    # 8) Return according to embedding type
    if embedding_type == "pheno_expr":
        logging.info(
            f"Loaded {dataset}: "
            f"{onehot.shape[0]} nodes, {edge_index.shape[1]} edges, "
            f"{onehot.shape[-1]} phenotype features, "
            f"{expr.shape[-1]} expression features"
        )
        return to_float_tensor(onehot), edge_index, y, adata, n_classes, to_float_tensor(expr)

    elif embedding_type == "pheno":
        logging.info(
            f"Loaded {dataset}: "
            f"{onehot.shape[0]} nodes, {edge_index.shape[1]} edges, "
            f"{onehot.shape[-1]} phenotype features"
        )
        return to_float_tensor(onehot), edge_index, y, adata, n_classes, None

    elif embedding_type == "expr":
        logging.info(
            f"Loaded {dataset}: "
            f"{expr.shape[0]} nodes, {edge_index.shape[1]} edges, "
            f"{expr.shape[-1]} expression features"
        )
        return to_float_tensor(expr), edge_index, y, adata, n_classes, None
    
    elif embedding_type == "embedding":
        logging.info(
            f"Loaded {dataset}: "
            f"{embedding.shape[0]} nodes, {edge_index.shape[1]} edges, "
            f"{embedding.shape[-1]} embedding features from adata.obsm['{embedding_key}']"
        )
        return to_float_tensor(embedding), edge_index, y, adata, n_classes, None

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
    km = KMeans(n_clusters=n_clusters, max_iter=5000, n_init=10)
    # km = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
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
            sil = silhouette_score(feats, true_labels)

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
    


def _rng(random_state: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy Generator with the requested seed (or global RNG)."""
    return np.random.default_rng(random_state)


def _nearest_neighbors(
    x: np.ndarray, k: int, **kwargs
) -> np.ndarray:
    """
    Return indices of the *k* nearest neighbours for every point in *x*.

    The first neighbour returned by ``sklearn`` is the query point itself,
    so we discard it.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, **kwargs).fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]  # drop self‑index
    return indices


def _encode_labels(labels: ArrayLike) -> np.ndarray:
    """
    Map arbitrary label values to consecutive integers starting from 0.

    This simplifies downstream use of ``np.bincount`` and avoids
    large sparse counts when label values are not contiguous.
    """
    labels = np.asarray(labels)
    _, encoded = np.unique(labels, return_inverse=True)
    return encoded


# ---------------------------------------------------------------------
# 1. Entropy of Batch Mixing
# ---------------------------------------------------------------------
def compute_entropy_batch_mixing(
    embeddings: np.ndarray,
    batch_labels: ArrayLike,
    k: int = 50,
    normalize: bool = True,
    **nn_kwargs,
) -> float:
    """
    Average entropy of batch labels in the *k*‑NN neighbourhood of each cell.

    Parameters
    ----------
    embeddings
        Low‑dimensional representation of shape *(n_cells, n_dims)*.
    batch_labels
        Iterable of length *n_cells* with one label per cell.
    k
        Number of neighbours (*excluding* the query cell) to consider.
    normalize
        If ``True`` (default) divide by the maximal entropy
        ``log(n_batches)``, yielding values in ``[0, 1]``.
    **nn_kwargs
        Additional arguments forwarded to :class:`sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    float
        Mean entropy across all cells.
    """
    if k < 1:
        raise ValueError("k must be ≥ 1")

    batch_labels = _encode_labels(batch_labels)
    indices = _nearest_neighbors(embeddings, k=k, **nn_kwargs)

    n_batches = int(batch_labels.max()) + 1
    max_ent = np.log(n_batches) if normalize else 1.0

    entropies = []
    for nbr in indices:
        counts = np.bincount(batch_labels[nbr], minlength=n_batches)
        probs = counts / k  # guaranteed non‑negative, summing to 1
        ent = entropy(probs) / max_ent if max_ent > 0 else 0.0
        entropies.append(ent)

    return float(np.mean(entropies))


# ---------------------------------------------------------------------
# 2. iLISI
# ---------------------------------------------------------------------
def compute_ilisi(
    embeddings: np.ndarray,
    batch_labels: ArrayLike,
    k: int = 90,
    **nn_kwargs,
) -> np.ndarray:
    """
    Compute the *inverse* Local Inverse Simpson’s Index (iLISI).

    For each cell *i*::

        iLISI_i = 1 − (# neighbours from same batch) / k

    Thus, 0 indicates perfect batch isolation; 1 indicates perfect mixing.

    Parameters
    ----------
    embeddings
        *(n_cells, n_dims)* array.
    batch_labels
        Iterable with one batch label per cell.
    k
        Number of neighbours (*excluding* the query cell) to consider.
    **nn_kwargs
        Extra arguments for :class:`sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    np.ndarray
        Vector of length *n_cells* with iLISI scores.
    """
    batch_labels = _encode_labels(batch_labels)
    indices = _nearest_neighbors(embeddings, k=k, **nn_kwargs)

    same_batch = (batch_labels[indices] == batch_labels[:, None]).sum(axis=1)
    ilisi = 1.0 - (same_batch / k)
    return ilisi


# ---------------------------------------------------------------------
# 3. Seurat Alignment Score (SAS)
# ---------------------------------------------------------------------
def compute_seurat_alignment_score(
    embeddings: np.ndarray,
    batch_labels: ArrayLike,
    neighbor_frac: float = 0.01,
    n_repeats: int = 3,
    random_state: Optional[int] = None,
    **nn_kwargs,
) -> float:
    """
    Seurat Alignment Score (Butler *et al.*, Cell 2018).

    The score estimates how well batches mix after integration by repeatedly
    down‑sampling to equal batch sizes and measuring the proportion of
    cross‑batch neighbours.

    Parameters
    ----------
    embeddings
        *(n_cells, n_dims)* array.
    batch_labels
        Iterable with one batch label per cell.
    neighbor_frac
        Fraction of the (sub‑sampled) cells to use as *k* in *k*‑NN.
        Must be in ``(0, 1]``.
    n_repeats
        Number of random sub‑samples to average.
    random_state
        Seed for reproducibility.
    **nn_kwargs
        Extra arguments for :class:`sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    float
        Mean SAS across repeats (1 = perfect mixing, 0 = no mixing).
    """
    if not (0 < neighbor_frac <= 1):
        raise ValueError("neighbor_frac must be in (0, 1]")

    rng = _rng(random_state)
    batch_labels = _encode_labels(batch_labels)
    batch_indices = [np.where(batch_labels == b)[0] for b in np.unique(batch_labels)]
    min_size = min(len(idx) for idx in batch_indices)
    n_batches = len(batch_indices)

    scores = []
    for _ in range(n_repeats):
        # balanced subsample
        sel = np.concatenate([rng.choice(idx, min_size, replace=False) for idx in batch_indices])
        x_sub, y_sub = embeddings[sel], batch_labels[sel]

        k = max(int(round(len(sel) * neighbor_frac)), 1)
        indices = _nearest_neighbors(x_sub, k=k, **nn_kwargs)

        same_batch = (y_sub[indices] == y_sub[:, None]).sum(axis=1).mean()
        score = (k - same_batch) * n_batches / (k * (n_batches - 1))
        scores.append(min(score, 1.0))  # numerical guard

    return float(np.mean(scores))


# ---------------------------------------------------------------------
# 4. ASW‑batch
# ---------------------------------------------------------------------
def compute_avg_silhouette_width_batch(
    embeddings: np.ndarray,
    batch_labels: ArrayLike,
    cell_types: ArrayLike,
    min_cells: int = 3,
    **silhouette_kwargs,
) -> float:
    """
    Average Silhouette Width computed per cell‑type, then averaged.

    Parameters
    ----------
    ...
    min_cells
        Minimum number of cells a cell‑type must have to be included.
    """
    x = embeddings
    y = _encode_labels(batch_labels)
    ct = _encode_labels(cell_types)

    scores = []
    for t in np.unique(ct):
        mask = ct == t
        n = mask.sum()
        n_labels = np.unique(y[mask]).size
        # Skip if too few cells OR labels≈cells (invalid for silhouette)
        if n < min_cells or n_labels < 2 or n_labels >= n:
            scores.append(0.0)
            continue
        try:
            s = silhouette_samples(x[mask], y[mask], **silhouette_kwargs)
            scores.append((1.0 - np.abs(s)).mean())
        except ValueError:
            scores.append(0.0)

    return float(np.mean(scores))

