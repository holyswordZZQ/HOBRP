import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

def _power_push_sor_core(adj_matrix: Dict[int, csr_matrix], indptr: Dict[int, np.ndarray], indices: Dict[int, np.ndarray], degree: Dict[int, np.ndarray], clique: Dict[int, List[List[int]]], clique_num: List[int], source: int, eps: float, alpha: float, omega: float, max_clique_size: int) -> Dict[int, np.ndarray]:
    n = clique_num[0]
    gpr_vec: Dict[int, np.ndarray] = {}
    for order in range(1, max_clique_size + 1):
        deg_s = degree[order][source]
        nbrs = indices[order][indptr[order][source]:indptr[order][source + 1]]
        if deg_s == 0 or (deg_s == 1 and len(nbrs) > 0 and (nbrs[0] == source)):
            gpr_vec[order] = np.zeros(n, dtype=np.float64)
            break
        if order == 1:
            queue_size = n
        else:
            queue_size = n + clique_num[order]
            high_degree = np.array([order + 1] * clique_num[order], dtype=np.int64)
            degree[order] = np.concatenate((degree[order], high_degree))
        queue = np.zeros(queue_size, dtype=np.int64)
        front, rear = (np.int64(0), np.int64(1))
        gpr_vec[order], residual_vec = (np.zeros(queue_size, dtype=np.float64), np.zeros(queue_size, dtype=np.float64))
        switch_size = np.int64(queue_size / 4)
        queue[rear] = source
        q_mark = np.zeros(queue_size, dtype=np.bool_)
        q_mark[source] = True
        residual_vec[source] = 1.0
        eps_cur = eps / max(float(indptr[order][-1]), 1.0)
        if clique_num[order] > 0:
            r_max = eps_cur / float(clique_num[order])
        else:
            r_max = eps_cur
        r_sum = 1.0
        eps_vec = r_max * degree[order]
        num_oper = 0.0
        step = 100000.0
        threshold = step
        while front != rear and rear - front <= switch_size:
            front = (front + 1) % queue_size
            u = queue[front]
            if not (order > 1 and u >= n):
                q_mark[u] = False
            if np.abs(residual_vec[u]) > eps_vec[u]:
                residual = omega * alpha * residual_vec[u]
                gpr_vec[order][u] += residual
                r_sum -= residual
                if degree[order][u] == 0:
                    if front == rear:
                        rear = (rear + 1) % queue_size
                        queue[rear] = source
                    continue
                increment = omega * (1.0 - alpha) * residual_vec[u] / degree[order][u]
                residual_vec[u] -= omega * residual_vec[u]
                num_oper += degree[order][u]
                if num_oper > threshold:
                    threshold += step
                    if threshold > 30000000.0:
                        step = 100000.0
                if order > 1 and u >= n:
                    u_real = u % n
                    for v in clique[order][u_real]:
                        residual_vec[v] += increment
                        if not q_mark[v]:
                            rear = (rear + 1) % queue_size
                            queue[rear] = v
                            q_mark[v] = True
                else:
                    for v in indices[order][indptr[order][u]:indptr[order][u + 1]]:
                        if order > 1:
                            v = n + v
                        residual_vec[v] += increment
                        if not q_mark[v]:
                            rear = (rear + 1) % queue_size
                            queue[rear] = v
                            q_mark[v] = True
        jump = r_sum <= eps_cur
        num_epoch = 8
        r_max_prime1 = np.power(eps_cur, 2.0 / num_epoch)
        degree_sum = float(np.sum(degree[order]))
        if degree_sum <= 0:
            r_max_prime2 = r_max_prime1
        else:
            r_max_prime2 = r_max_prime1 / degree_sum
        if not jump:
            for _ in np.arange(1, num_epoch + 1):
                while r_sum > r_max_prime1:
                    for u in range(queue_size):
                        if r_sum <= r_max_prime1:
                            jump = True
                            break
                        if np.abs(residual_vec[u]) > r_max_prime2 * degree[order][u]:
                            residual = omega * alpha * residual_vec[u]
                            gpr_vec[order][u] += residual
                            r_sum -= residual
                            if degree[order][u] == 0:
                                continue
                            increment = omega * (1.0 - alpha) * residual_vec[u] / degree[order][u]
                            residual_vec[u] -= omega * residual_vec[u]
                            num_oper += degree[order][u]
                            if num_oper > threshold:
                                threshold += step
                                if threshold > 30000000.0:
                                    step = 100000.0
                            if order > 1 and u >= n:
                                u_real = u % n
                                for v in clique[order][u_real]:
                                    residual_vec[v] += increment
                            else:
                                for index in indices[order][indptr[order][u]:indptr[order][u + 1]]:
                                    if order > 1:
                                        index = index + n
                                    residual_vec[index] += increment
    for k in list(gpr_vec.keys()):
        gpr_vec[k] = gpr_vec[k][:n]
    return gpr_vec

def power_push_sor(adj_matrix: Dict[int, csr_matrix], clique: Dict[int, List[List[int]]], clique_num: List[int], source: int, eps: float, alpha: float, omega: Optional[float]=None, max_clique_size: int=2) -> Dict[int, np.ndarray]:
    if omega is None:
        omega = 1.0 + ((1.0 - alpha) / (1.0 + np.sqrt(1 - (1.0 - alpha) ** 2.0))) ** 2.0
    indices: Dict[int, np.ndarray] = {}
    indptr: Dict[int, np.ndarray] = {}
    degree: Dict[int, np.ndarray] = {}
    for i in range(1, max_clique_size + 1):
        indices[i] = adj_matrix[i].indices
        indptr[i] = adj_matrix[i].indptr
        degree[i] = np.int64(adj_matrix[i].sum(1).A.flatten())
    return _power_push_sor_core(adj_matrix=adj_matrix, indptr=indptr, indices=indices, degree=degree, clique=clique, clique_num=clique_num, source=source, eps=eps, alpha=alpha, omega=omega, max_clique_size=max_clique_size)

def _str2bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise ValueError(f'Invalid boolean value: {value}')

def _sorted_fold_dirs(split_root: Path) -> List[Path]:
    fold_dirs = []
    for item in split_root.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if not name.startswith('fold_'):
            continue
        try:
            fold_id = int(name.split('_', 1)[1])
        except (IndexError, ValueError):
            continue
        fold_dirs.append((fold_id, item))
    fold_dirs.sort(key=lambda x: x[0])
    return [x[1] for x in fold_dirs]

def _load_num_nodes(split_root: Path, fold_dirs: List[Path]) -> Tuple[int, int]:
    meta_path = split_root / 'meta.json'
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        num_drug = int(meta['num_drug'])
        num_target = int(meta['num_target'])
        return (num_drug, num_target)
    max_drug = -1
    max_target = -1
    for fold_dir in fold_dirs:
        for file_name in ('train_pos_edges.csv', 'test_pos_edges.csv'):
            path = fold_dir / file_name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if 'drug_idx' not in df.columns or 'target_idx' not in df.columns:
                continue
            if not df.empty:
                max_drug = max(max_drug, int(df['drug_idx'].max()))
                max_target = max(max_target, int(df['target_idx'].max()))
    if max_drug < 0 or max_target < 0:
        raise ValueError(f'Cannot infer num_drug/num_target from {split_root}. Provide a split root containing meta.json or valid edge CSVs.')
    return (max_drug + 1, max_target + 1)

def _load_train_edge_index(train_csv: Path, num_drug: int, num_target: int, add_self_loops: bool) -> Tuple[np.ndarray, np.ndarray, int]:
    df = pd.read_csv(train_csv)
    if 'drug_idx' not in df.columns or 'target_idx' not in df.columns:
        raise ValueError(f'Missing drug_idx/target_idx columns in {train_csv}')
    if df.empty:
        raise ValueError(f'Empty training edge file: {train_csv}')
    drug_idx = df['drug_idx'].astype(np.int64).to_numpy()
    target_idx = df['target_idx'].astype(np.int64).to_numpy()
    if drug_idx.min() < 0 or drug_idx.max() >= num_drug:
        raise ValueError(f'drug_idx out of range in {train_csv}. Expected [0, {num_drug - 1}], got [{drug_idx.min()}, {drug_idx.max()}].')
    if target_idx.min() < 0 or target_idx.max() >= num_target:
        raise ValueError(f'target_idx out of range in {train_csv}. Expected [0, {num_target - 1}], got [{target_idx.min()}, {target_idx.max()}].')
    target_global = target_idx + num_drug
    src = np.concatenate([drug_idx, target_global]).astype(np.int64)
    dst = np.concatenate([target_global, drug_idx]).astype(np.int64)
    if add_self_loops:
        num_nodes = num_drug + num_target
        self_nodes = np.arange(num_nodes, dtype=np.int64)
        src = np.concatenate([src, self_nodes])
        dst = np.concatenate([dst, self_nodes])
    pairs = np.stack([src, dst], axis=1)
    pairs = np.unique(pairs, axis=0)
    src_unique = pairs[:, 0]
    dst_unique = pairs[:, 1]
    num_train_edges = int(df.shape[0])
    return (src_unique, dst_unique, num_train_edges)

def _prepare_higher_order_structures(src: np.ndarray, dst: np.ndarray, num_nodes: int, order: int) -> Tuple[Dict[int, csr_matrix], Dict[int, List[List[int]]], List[int]]:
    row: Dict[int, List[int]] = {}
    col: Dict[int, List[int]] = {}
    clique: Dict[int, List[List[int]]] = {}
    adj_matrix_csr: Dict[int, csr_matrix] = {}
    clique_num = [0] * (order + 1)
    for i in range(1, order + 1):
        row[i], col[i], clique[i] = ([], [], [])
    row[1] = src.tolist()
    col[1] = dst.tolist()
    clique_num[0] = num_nodes
    clique_num[1] = len(row[1])
    if order > 1:
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(np.stack([src, dst], axis=1).tolist())
        for clique_nodes in nx.enumerate_all_cliques(graph):
            clique_size = len(clique_nodes)
            if clique_size <= 2:
                continue
            if clique_size > order + 1:
                break
            tmp_order = clique_size - 1
            clique[tmp_order].append(clique_nodes)
            row[tmp_order].extend(clique_nodes)
            col[tmp_order].extend([clique_num[tmp_order]] * clique_size)
            clique_num[tmp_order] += 1
    for i in range(1, order + 1):
        if len(row[i]) == 0:
            if i == 1:
                adj_matrix_csr[i] = csr_matrix((num_nodes, num_nodes), dtype=np.float64)
            else:
                adj_matrix_csr[i] = csr_matrix((num_nodes, clique_num[i]), dtype=np.float64)
            continue
        data_values = np.ones(len(row[i]), dtype=np.float64)
        if i == 1:
            adj_matrix_csr[i] = csr_matrix((data_values, (row[i], col[i])), shape=(clique_num[0], clique_num[0]))
        else:
            adj_matrix_csr[i] = csr_matrix((data_values, (row[i], col[i])), shape=(clique_num[0], clique_num[i]))
    return (adj_matrix_csr, clique, clique_num)

def _build_relation_operator_for_fold(src: np.ndarray, dst: np.ndarray, num_nodes: int, order: int, eps: float, alpha: float, show_progress: bool) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    adj_matrix_csr, clique, clique_num = _prepare_higher_order_structures(src=src, dst=dst, num_nodes=num_nodes, order=order)
    all_nonzero_values: Dict[int, List[torch.Tensor]] = {}
    all_col_indices: Dict[int, List[torch.Tensor]] = {}
    all_row_indices: Dict[int, List[torch.Tensor]] = {}
    for i in range(1, order + 1):
        all_nonzero_values[i], all_col_indices[i], all_row_indices[i] = ([], [], [])
    node_iter = range(num_nodes)
    if show_progress:
        node_iter = tqdm(node_iter, desc='Computing relation operator', leave=False)
    for node in node_iter:
        restart_scores = power_push_sor(adj_matrix=adj_matrix_csr, clique=clique, clique_num=clique_num, source=node, eps=eps, alpha=alpha, max_clique_size=order)
        for i in range(1, order + 1):
            cur = restart_scores.get(i)
            if cur is None or float(np.sum(cur)) == 0.0:
                break
            restart_scores_tensor = torch.tensor(cur, dtype=torch.float64)
            col_indices = torch.nonzero(restart_scores_tensor > 0, as_tuple=False).squeeze()
            nonzero_values = restart_scores_tensor[col_indices]
            row_indices = torch.full_like(col_indices, node)
            if nonzero_values.dim() == 0:
                nonzero_values = nonzero_values.unsqueeze(0)
            if col_indices.dim() == 0:
                col_indices = col_indices.unsqueeze(0)
            if row_indices.dim() == 0:
                row_indices = row_indices.unsqueeze(0)
            all_nonzero_values[i].append(nonzero_values)
            all_col_indices[i].append(col_indices)
            all_row_indices[i].append(row_indices)
    relation_operator: Dict[int, torch.Tensor] = {}
    relation_operator_nnz: Dict[int, int] = {}
    for i in range(1, order + 1):
        if all_nonzero_values[i]:
            cur_values = torch.cat(all_nonzero_values[i])
            cur_cols = torch.cat(all_col_indices[i])
            cur_rows = torch.cat(all_row_indices[i])
            coo_indices = torch.stack([cur_rows, cur_cols], dim=0)
            sparse = torch.sparse_coo_tensor(coo_indices, cur_values, size=torch.Size([num_nodes, num_nodes]), dtype=torch.float64).coalesce()
        else:
            sparse = torch.sparse_coo_tensor(indices=torch.empty((2, 0), dtype=torch.long), values=torch.empty((0,), dtype=torch.float64), size=torch.Size([num_nodes, num_nodes]), dtype=torch.float64).coalesce()
        relation_operator[i] = sparse
        relation_operator_nnz[i] = int(sparse._nnz())
    return (relation_operator, relation_operator_nnz)

def run(split_root: str, order: int=4, eps: float=1e-08, alpha: float=0.15, overwrite: bool=False, add_self_loops: bool=False, folds: Optional[List[int]]=None, summary_name: str='train_relation_operator_summary.json', show_progress: bool=True) -> Dict:
    split_root_path = Path(split_root).resolve()
    if not split_root_path.exists():
        raise FileNotFoundError(f'split_root not found: {split_root_path}')
    if order < 1:
        raise ValueError(f'order must be >= 1, got {order}')
    fold_dirs = _sorted_fold_dirs(split_root_path)
    if not fold_dirs:
        raise ValueError(f'No fold_* directories found under {split_root_path}')
    if folds:
        fold_set = set((int(x) for x in folds))
        filtered = []
        for fold_dir in fold_dirs:
            fold_id = int(fold_dir.name.split('_', 1)[1])
            if fold_id in fold_set:
                filtered.append(fold_dir)
        fold_dirs = filtered
        if not fold_dirs:
            raise ValueError(f'No matching folds found for {sorted(fold_set)}')
    num_drug, num_target = _load_num_nodes(split_root=split_root_path, fold_dirs=fold_dirs)
    num_nodes = num_drug + num_target
    summary_folds = []
    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split('_', 1)[1])
        train_path = fold_dir / 'train_pos_edges.csv'
        if not train_path.exists():
            raise FileNotFoundError(f'Missing training edge file: {train_path}')
        out_paths = {i: fold_dir / f'train_relation_operator_order_{i}.pt' for i in range(1, order + 1)}
        if not overwrite and all((path.exists() for path in out_paths.values())):
            relation_operator_nnz = {}
            for i, path in out_paths.items():
                sp_tensor = torch.load(path, map_location='cpu')
                if not isinstance(sp_tensor, torch.Tensor) or not sp_tensor.is_sparse:
                    raise ValueError(f'Existing file is not sparse tensor: {path}')
                relation_operator_nnz[i] = int(sp_tensor._nnz())
            summary_folds.append({'fold': fold_id, 'fold_dir': str(fold_dir), 'skipped': True, 'reason': 'all train_relation_operator_order_*.pt already exist', 'saved_relation_operator_orders': list(range(1, order + 1)), 'saved_paths': [str(out_paths[i]) for i in range(1, order + 1)], 'relation_operator_nnz': {str(k): int(v) for k, v in relation_operator_nnz.items()}})
            continue
        src, dst, num_train_edges = _load_train_edge_index(train_csv=train_path, num_drug=num_drug, num_target=num_target, add_self_loops=add_self_loops)
        relation_operator, relation_operator_nnz = _build_relation_operator_for_fold(src=src, dst=dst, num_nodes=num_nodes, order=order, eps=eps, alpha=alpha, show_progress=show_progress)
        for i in range(1, order + 1):
            torch.save(relation_operator[i], out_paths[i])
        summary_folds.append({'fold': fold_id, 'fold_dir': str(fold_dir), 'num_train_edges': int(num_train_edges), 'edge_index_nnz': int(len(src)), 'saved_relation_operator_orders': list(range(1, order + 1)), 'saved_paths': [str(out_paths[i]) for i in range(1, order + 1)], 'relation_operator_nnz': {str(k): int(v) for k, v in relation_operator_nnz.items()}})
    summary = {'split_root': str(split_root_path), 'num_drug': int(num_drug), 'num_target': int(num_target), 'num_nodes': int(num_nodes), 'order': int(order), 'eps': float(eps), 'restart_probability': float(alpha), 'add_self_loops': bool(add_self_loops), 'overwrite': bool(overwrite), 'num_folds': int(len(summary_folds)), 'folds': summary_folds}
    summary_path = split_root_path / summary_name
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

def _parse_fold_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    out = []
    for part in value.split(','):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out if out else None
