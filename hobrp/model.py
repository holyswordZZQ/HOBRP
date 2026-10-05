import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import json
import os
import numpy as np
from types import SimpleNamespace
from torch.nn import Linear, Parameter

def _relation_operator_matmul(relation_operator, x):
    if isinstance(relation_operator, torch.Tensor):
        if relation_operator.layout != torch.strided:
            if relation_operator.layout == torch.sparse_coo:
                relation_operator_sp = relation_operator.coalesce()
            else:
                relation_operator_sp = relation_operator.to_sparse_coo().coalesce()
            relation_operator_sp = relation_operator_sp.to(device=x.device, dtype=x.dtype)
            return torch.sparse.mm(relation_operator_sp, x)
        return torch.matmul(relation_operator.to(device=x.device, dtype=x.dtype), x)
    to_coo = getattr(relation_operator, 'to_torch_sparse_coo_tensor', None)
    if callable(to_coo):
        relation_operator_coo = to_coo()
        if not isinstance(relation_operator_coo, torch.Tensor):
            raise TypeError('to_torch_sparse_coo_tensor() must return torch.Tensor')
        relation_operator_coo = relation_operator_coo.coalesce().to(device=x.device, dtype=x.dtype)
        return torch.sparse.mm(relation_operator_coo, x)
    mm = getattr(relation_operator, 'matmul', None)
    if callable(mm):
        return mm(x)
    raise TypeError(f'Unsupported operator type: {type(relation_operator)}')

class RelationPropagationBranch(nn.Module):

    def __init__(self, K, alpha, relation_order=2):
        super().__init__()
        self.K = int(K)
        self.alpha = float(alpha)
        self.relation_order = int(relation_order)
        self.fW = Parameter(torch.empty(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.fW.zero_()
            for k in range(self.K + 1):
                self.fW[k] = self.alpha * (1.0 - self.alpha) ** k
            self.fW[-1] = (1.0 - self.alpha) ** self.K

    def forward(self, x, relation_operator):
        hidden = x * self.fW[0]
        for k in range(self.K):
            x = _relation_operator_matmul(relation_operator, x)
            hidden = hidden + self.fW[k + 1] * x
        return hidden

    def __repr__(self):
        return f'{self.__class__.__name__}(relation_order={self.relation_order}, K={self.K}, filterWeights={self.fW})'

class HOBRPLinkPredictor(nn.Module):

    def __init__(self, num_nodes, node_features=None, emb_dim=128, hidden_dim=128, dropout=0.2, K=10, alpha=0.1, restart_probability=0.15, eps=1e-08, order=2, dprate=0.0, relation_operator_cache_dir=None, edge_hidden_dim=None):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.order = int(order)
        self.K = int(K)
        self.alpha = float(alpha)
        self.restart_probability = float(restart_probability)
        self.eps = float(eps)
        self.dropout = float(dropout)
        self.dprate = float(dprate)
        self.relation_operator_cache_dir = relation_operator_cache_dir
        self.precomputed_relation_operator = None
        if node_features is not None:
            if node_features.size(0) != self.num_nodes:
                raise ValueError('node_features row count must equal num_nodes')
            if node_features.size(1) != int(emb_dim):
                raise ValueError('node_features column count must equal emb_dim')
            self.register_buffer('node_features', node_features.detach().clone())
        else:
            self.node_features = None
        self.lin_in = nn.ModuleList([Linear(int(emb_dim), int(hidden_dim)) for _ in range(self.order)])
        self.propagation_branches = nn.ModuleList([RelationPropagationBranch(self.K, self.alpha, self.order) for _ in range(self.order)])
        self.lin_out = Linear(int(hidden_dim) * self.order, int(hidden_dim))
        edge_hidden_dim = int(edge_hidden_dim) if edge_hidden_dim is not None else int(hidden_dim)
        self.edge_mlp = nn.Sequential(nn.Linear(int(hidden_dim) * 2, edge_hidden_dim), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(edge_hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.lin_in:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for prop in self.propagation_branches:
            prop.reset_parameters()
        nn.init.xavier_uniform_(self.lin_out.weight)
        if self.lin_out.bias is not None:
            nn.init.zeros_(self.lin_out.bias)
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def set_precomputed_relation_operator(self, relation_operator):
        self.precomputed_relation_operator = relation_operator

    def _resolve_branch_relation_operator(self, relation_operator_container, order_idx):
        if isinstance(relation_operator_container, dict):
            key_candidates = [order_idx, str(order_idx), order_idx - 1, str(order_idx - 1), f'order_{order_idx}', f'relation_operator_order_{order_idx}']
            for key in key_candidates:
                if key in relation_operator_container:
                    return relation_operator_container[key]
            if len(relation_operator_container) == 1:
                return next(iter(relation_operator_container.values()))
            raise KeyError(f'Cannot find operator branch for order={order_idx}. Available keys: {list(relation_operator_container.keys())}')
        if isinstance(relation_operator_container, (list, tuple)):
            if len(relation_operator_container) == 0:
                raise ValueError('operator list/tuple is empty')
            if len(relation_operator_container) == 1:
                return relation_operator_container[0]
            idx = order_idx - 1
            if idx < len(relation_operator_container):
                return relation_operator_container[idx]
            return relation_operator_container[-1]
        return relation_operator_container

    def _resolve_x_and_relation_operator(self, data_or_edge_index=None, relation_operator=None):
        x = None
        relation_operator_container = relation_operator
        if hasattr(data_or_edge_index, 'x'):
            x = data_or_edge_index.x
        if hasattr(data_or_edge_index, 'operator') and relation_operator_container is None:
            relation_operator_container = data_or_edge_index.operator
        if x is None:
            if self.node_features is None:
                raise ValueError('No node features found. Provide data.x or initialize model with node_features.')
            x = self.node_features
        if relation_operator_container is None:
            relation_operator_container = self.precomputed_relation_operator
        if relation_operator_container is None:
            raise ValueError('No operator provided. Pass data.operator / relation_operator=... / set_precomputed_relation_operator(...). You can load Data/operators/*/relation_operator_order_2.pt and pass it as a single operator tensor.')
        return (x, relation_operator_container)

    def encode(self, data_or_edge_index=None, relation_operator=None):
        x, relation_operator_container = self._resolve_x_and_relation_operator(data_or_edge_index=data_or_edge_index, relation_operator=relation_operator)
        branch_outputs = []
        for i in range(self.order):
            order_idx = i + 1
            branch_relation_operator = self._resolve_branch_relation_operator(relation_operator_container, order_idx)
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.propagation_branches[i](xx, branch_relation_operator)
            branch_outputs.append(xx)
        x_concat = torch.cat(branch_outputs, dim=1)
        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        return self.lin_out(x_concat)

    @staticmethod
    def _edge_pairs_to_tensor(edge_pairs, device):
        if isinstance(edge_pairs, torch.Tensor):
            t = edge_pairs.to(device=device, dtype=torch.long)
            if t.dim() != 2:
                raise ValueError('edge_pairs tensor must be 2D')
            if t.size(1) == 2:
                return t
            if t.size(0) == 2:
                return t.t().contiguous()
            raise ValueError(f'Invalid edge_pairs tensor shape: {tuple(t.shape)}')
        return torch.tensor(edge_pairs, dtype=torch.long, device=device)

    def decode(self, z, edge_pairs, num_drug, edge_pairs_are_global=False):
        pair_tensor = self._edge_pairs_to_tensor(edge_pairs=edge_pairs, device=z.device)
        if pair_tensor.numel() == 0:
            return torch.empty((0,), dtype=z.dtype, device=z.device)
        d_idx = pair_tensor[:, 0]
        if edge_pairs_are_global:
            t_idx = pair_tensor[:, 1]
        else:
            t_idx = pair_tensor[:, 1] + int(num_drug)
        zd = z[d_idx]
        zt = z[t_idx]
        return self.edge_mlp(torch.cat([zd, zt], dim=1)).squeeze(1)

    def forward(self, data_or_edge_index, edge_pairs, num_drug, device=None, relation_operator=None, edge_pairs_are_global=False):
        del device
        z = self.encode(data_or_edge_index=data_or_edge_index, relation_operator=relation_operator)
        return self.decode(z=z, edge_pairs=edge_pairs, num_drug=num_drug, edge_pairs_are_global=edge_pairs_are_global)

def build_model(**kwargs):
    return HOBRPLinkPredictor(**kwargs)
