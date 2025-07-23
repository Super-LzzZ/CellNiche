from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import os
import pickle
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from typing import Union, List, Optional, Callable
import numpy as np

from node2vec import Node2Vec
import networkx as nx


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class NeighborSampler(torch.utils.data.DataLoader):
    r"""
    This code adapted from
    (https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/neighbor_sampler.py).
    (https://github.com/EdisonLeeeee/MAGI/blob/master/magi/neighbor_sampler.py).
    """
    def __init__(self,
                 edge_index: Union[Tensor, SparseTensor],
                 adj: SparseTensor,
                 sizes: List[int],
                 is_train: bool = False,
                 wt: int = 20,
                 wl: int = 4,
                 biased_random_walk: bool = False,
                 p: float = 1.0,
                 q: float = 1.0,
                 drop_last= False,
                 node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None,
                 return_e_id: bool = True,
                 transform: Callable = None,
                 sim_adj: SparseTensor = None,
                 **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning < 1.6:
        self.edge_index = edge_index
        self.adj = adj
        self.sim_adj = None
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        self.is_train = is_train
        self.drop_last = drop_last
        self.wt = wt
        self.wl = wl
        self.biased_random_walk = biased_random_walk
        self.p = p
        self.q = q

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, drop_last=self.drop_last, **kwargs)

    def _get_pos_neg_batch(self, random_nodes, biased_random_walk):
        random_nodes_count = random_nodes.shape[0]
        rowptr, col, _ = self.adj.csr()

        # -------------------------------
        # Stage 1: Full-graph random walk
        # -------------------------------
        random_nodes_repeat = random_nodes.repeat(self.wt)
        rw1 = self.adj.random_walk(random_nodes_repeat, self.wl)[:, 1:]
        # print(f"rw1.shape: {rw1.shape}")
        if not isinstance(rw1, torch.Tensor):
            rw1 = rw1[0]
        rw1 = rw1.t().reshape(-1, random_nodes_count).t()
        batch = []
        for i in range(random_nodes_count):
            rw_nodes, rw_times = torch.unique(rw1[i], return_counts=True)
            nodes = rw_nodes[rw_times > rw_times.float().mean()].tolist()
            batch += nodes
        batch += random_nodes.tolist()
        batch = torch.tensor(batch).unique()

        # -------------------------------
        # Stage 2: Biased/Unbiased random walk
        # -------------------------------
        batch_size = batch.shape[0]
        if biased_random_walk == True:
            walks_file = os.path.join("./results/walsk/", "node2vec_walks.pkl")
            if os.path.exists(walks_file):
                print(f"Found precomputed walks at {walks_file}, loading...")
                with open(walks_file, "rb") as f:
                    walks = pickle.load(f)
            else:
                print(f"No precomputed walks found. Computing full graph walks...")
                full_graph = self._create_graph_from_adj(self.adj)
                node2vec = Node2Vec(full_graph, walk_length=self.wl, num_walks=self.wt, p=self.p, q=self.q, workers=8)
                walks = node2vec.walks
                with open(walks_file, "wb") as f:
                    pickle.dump(walks, f)
                print(f"Walks saved to {walks_file}")

            node2vec_walks = [list(map(int, walk)) for walk in walks]
            rw2 = torch.tensor(node2vec_walks, dtype=torch.long)
            batch_set = set(batch.tolist())
            mask = []
            for i in range(rw2.size(0)):
                row_nodes = rw2[i].tolist()
                if row_nodes[0] in batch_set:
                    mask.append(True)
                else:
                    mask.append(False)
            mask = torch.tensor(mask, dtype=torch.bool)
            rw2 = rw2[mask]

        else:
            batch_repeat = batch.repeat(self.wt)
            rw2 = self.adj.random_walk(batch_repeat, self.wl)[:, 1:]
            if not isinstance(rw2, torch.Tensor):
                rw2 = rw2[0]

        rw2 = rw2.t().reshape(-1, batch_size).t()
        row, col, val = [], [], []
        for i in range(batch.shape[0]):
            rw2_nodes, rw2_times = torch.unique(rw2[i], return_counts=True)
            row += [batch[i].item()] * rw2_nodes.shape[0]
            col += rw2_nodes.tolist()
            val += rw2_times.tolist()

        unique_nodes = list(set(row + col))
        subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

        row = [subg2g[x] for x in row]
        col = [subg2g[x] for x in col]
        idx = torch.tensor([subg2g[x] for x in batch.tolist()])

        adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                            sparse_sizes=(len(unique_nodes), len(unique_nodes)))

        adj_batch, _ = adj_.saint_subgraph(idx)
        adj_batch_sp = adj_batch.to_scipy(layout='coo')
        adj_batch_sp.setdiag([0] * idx.shape[0])
        adj_batch = SparseTensor.from_scipy(adj_batch_sp)

        if self.sim_adj is not None:
            sim_row, sim_col, sim_val = [], [], []
            for r, c in zip(row, col):
                sim_row.append(r)
                sim_col.append(c)

                adj_row, adj_col, adj_val = self.sim_adj.storage.row(), self.sim_adj.storage.col(), self.sim_adj.storage.value()
                mask = (adj_row == r) & (adj_col == c)
                if mask.any():
                    similarity = adj_val[mask].item()
                else:
                    similarity = 0.0 

                sim_val.append(similarity)

            sim_val_tensor = torch.tensor(sim_val, dtype=torch.float32)
            sim_adj_ = SparseTensor(row=torch.LongTensor(sim_row), col=torch.LongTensor(sim_col), value=sim_val_tensor,
                                    sparse_sizes=(len(unique_nodes), len(unique_nodes)))
            sim_adj_batch, _ = sim_adj_.saint_subgraph(idx)
            sim_adj_batch_sp = sim_adj_batch.to_scipy(layout='coo')
            sim_adj_batch_sp.setdiag([0] * idx.shape[0])
            sim_adj_batch = SparseTensor.from_scipy(sim_adj_batch_sp)

            return batch, adj_batch, sim_adj_batch

        return batch, adj_batch

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        adj_batch = None
        sim_adj_batch = None
        if self.sim_adj is not None:
            batch, adj_batch, sim_adj_batch = self._get_pos_neg_batch(batch, biased_random_walk=self.biased_random_walk)
        else:
            batch, adj_batch = self._get_pos_neg_batch(batch, biased_random_walk=self.biased_random_walk)
        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out

        if sim_adj_batch is not None:
            return out, adj_batch, sim_adj_batch, batch
        else:
            return out, adj_batch, batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'
