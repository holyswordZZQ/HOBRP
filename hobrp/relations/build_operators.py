import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm
from .restart_propagation import power_push_sor

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

def _load_node_features(node_feature_path: Path, num_drug: int, num_target: int) -> np.ndarray:
    data = np.load(str(node_feature_path), allow_pickle=True)
    expected_nodes = int(num_drug + num_target)
    if 'node_features' in data:
        node_features = data['node_features'].astype(np.float32)
        if node_features.shape[0] != expected_nodes:
            raise ValueError(f'node_features rows mismatch: got {node_features.shape[0]}, expected {expected_nodes}')
        return node_features
    if 'drug_features' in data and 'target_features' in data:
        drug_features = data['drug_features'].astype(np.float32)
        target_features = data['target_features'].astype(np.float32)
        if drug_features.shape[0] != num_drug or target_features.shape[0] != num_target:
            raise ValueError(f'drug_features/target_features rows mismatch: drug {drug_features.shape[0]} vs {num_drug}, target {target_features.shape[0]} vs {num_target}')
        return np.concatenate([drug_features, target_features], axis=0).astype(np.float32)
    raise ValueError(f'Cannot load node features from {node_feature_path}. Need key `node_features` or both `drug_features` and `target_features`.')

def _l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return features / norms

def _build_similarity_edges(node_features: np.ndarray, sim_topk: int, sim_min: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    num_nodes = int(node_features.shape[0])
    if num_nodes <= 1 or int(sim_topk) <= 0:
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64))
    topk = int(min(max(1, int(sim_topk)), num_nodes - 1))
    normalized = _l2_normalize_rows(node_features.astype(np.float32))
    sim = np.matmul(normalized, normalized.T).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    candidate_cols = np.argpartition(-sim, kth=topk - 1, axis=1)[:, :topk]
    candidate_scores = np.take_along_axis(sim, candidate_cols, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    topk_cols = np.take_along_axis(candidate_cols, order, axis=1)
    topk_scores = np.take_along_axis(candidate_scores, order, axis=1)
    src = np.repeat(np.arange(num_nodes, dtype=np.int64), topk)
    dst = topk_cols.reshape(-1).astype(np.int64)
    if sim_min is not None:
        mask = topk_scores.reshape(-1) >= float(sim_min)
        src = src[mask]
        dst = dst[mask]
    return (src, dst)

def _load_train_dti_edges(train_csv: Path, num_drug: int, num_target: int) -> Tuple[np.ndarray, np.ndarray, int]:
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
    target_global = target_idx + int(num_drug)
    src = np.concatenate([drug_idx, target_global]).astype(np.int64)
    dst = np.concatenate([target_global, drug_idx]).astype(np.int64)
    return (src, dst, int(df.shape[0]))

def _merge_edges_with_similarity(dti_src: np.ndarray, dti_dst: np.ndarray, sim_src: np.ndarray, sim_dst: np.ndarray, num_drug: int, num_target: int, add_self_loops: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    num_nodes = int(num_drug + num_target)
    src = np.concatenate([dti_src.astype(np.int64), sim_src.astype(np.int64)])
    dst = np.concatenate([dti_dst.astype(np.int64), sim_dst.astype(np.int64)])
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes) & (src != dst)
    src = src[valid]
    dst = dst[valid]
    if add_self_loops:
        self_nodes = np.arange(num_nodes, dtype=np.int64)
        src = np.concatenate([src, self_nodes])
        dst = np.concatenate([dst, self_nodes])
    pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
    src_unique = pairs[:, 0]
    dst_unique = pairs[:, 1]
    src_is_drug = src_unique < num_drug
    dst_is_drug = dst_unique < num_drug
    dd = int(np.sum(src_is_drug & dst_is_drug))
    tt = int(np.sum(~src_is_drug & ~dst_is_drug))
    dt = int(np.sum(src_is_drug ^ dst_is_drug))
    edge_stats = {'num_edges_dti_directed': int(len(dti_src)), 'num_edges_similarity_directed': int(len(sim_src)), 'num_edges_unique_directed': int(len(src_unique)), 'num_edges_unique_dd': dd, 'num_edges_unique_tt': tt, 'num_edges_unique_dt': dt}
    return (src_unique, dst_unique, edge_stats)

def _prepare_higher_order_structures_with_filter(src: np.ndarray, dst: np.ndarray, num_nodes: int, num_drug: int, order: int, remove_homo_cliques: bool) -> Tuple[Dict[int, csr_matrix], Dict[int, List[List[int]]], List[int]]:
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
            if remove_homo_cliques:
                node_types = np.array(clique_nodes, dtype=np.int64) < int(num_drug)
                if bool(np.all(node_types)) or bool(np.all(~node_types)):
                    continue
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

def _build_relation_operator_for_fold(src: np.ndarray, dst: np.ndarray, num_nodes: int, num_drug: int, order: int, eps: float, alpha: float, remove_homo_cliques: bool, show_progress: bool) -> Tuple[Dict[int, torch.Tensor], Dict[int, int], Dict[int, int]]:
    adj_matrix_csr, clique, clique_num = _prepare_higher_order_structures_with_filter(src=src, dst=dst, num_nodes=num_nodes, num_drug=num_drug, order=order, remove_homo_cliques=remove_homo_cliques)
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
    clique_count = {i: int(clique_num[i]) for i in range(2, order + 1)}
    return (relation_operator, relation_operator_nnz, clique_count)

def run(split_root: str, node_feature_path: str, order: int=4, eps: float=1e-08, alpha: float=0.15, sim_topk: int=10, sim_min: Optional[float]=None, overwrite: bool=False, add_self_loops: bool=False, remove_homo_cliques: bool=True, folds: Optional[List[int]]=None, output_prefix: str='relation_operator_order', summary_name: str='relation_operator_summary.json', show_progress: bool=True) -> Dict:
    split_root_path = Path(split_root).resolve()
    if not split_root_path.exists():
        raise FileNotFoundError(f'split_root not found: {split_root_path}')
    if order < 1:
        raise ValueError(f'order must be >= 1, got {order}')
    node_feature_path_obj = Path(node_feature_path).resolve()
    if not node_feature_path_obj.exists():
        raise FileNotFoundError(f'node_feature_path not found: {node_feature_path_obj}')
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
    node_features = _load_node_features(node_feature_path=node_feature_path_obj, num_drug=num_drug, num_target=num_target)
    sim_src, sim_dst = _build_similarity_edges(node_features=node_features, sim_topk=sim_topk, sim_min=sim_min)
    summary_folds = []
    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split('_', 1)[1])
        train_path = fold_dir / 'train_pos_edges.csv'
        if not train_path.exists():
            raise FileNotFoundError(f'Missing training edge file: {train_path}')
        out_paths = {i: fold_dir / f'{output_prefix}_{i}.pt' for i in range(1, order + 1)}
        if not overwrite and all((path.exists() for path in out_paths.values())):
            relation_operator_nnz = {}
            for i, path in out_paths.items():
                sp_tensor = torch.load(path, map_location='cpu')
                if not isinstance(sp_tensor, torch.Tensor) or not sp_tensor.is_sparse:
                    raise ValueError(f'Existing file is not sparse tensor: {path}')
                relation_operator_nnz[i] = int(sp_tensor._nnz())
            summary_folds.append({'fold': fold_id, 'fold_dir': str(fold_dir), 'skipped': True, 'reason': f'all {output_prefix}_*.pt already exist', 'saved_relation_operator_orders': list(range(1, order + 1)), 'saved_paths': [str(out_paths[i]) for i in range(1, order + 1)], 'relation_operator_nnz': {str(k): int(v) for k, v in relation_operator_nnz.items()}})
            continue
        dti_src, dti_dst, num_train_edges = _load_train_dti_edges(train_csv=train_path, num_drug=num_drug, num_target=num_target)
        src, dst, edge_stats = _merge_edges_with_similarity(dti_src=dti_src, dti_dst=dti_dst, sim_src=sim_src, sim_dst=sim_dst, num_drug=num_drug, num_target=num_target, add_self_loops=add_self_loops)
        relation_operator, relation_operator_nnz, clique_count = _build_relation_operator_for_fold(src=src, dst=dst, num_nodes=num_nodes, num_drug=num_drug, order=order, eps=eps, alpha=alpha, remove_homo_cliques=remove_homo_cliques, show_progress=show_progress)
        for i in range(1, order + 1):
            torch.save(relation_operator[i], out_paths[i])
        summary_folds.append({'fold': fold_id, 'fold_dir': str(fold_dir), 'num_train_edges': int(num_train_edges), 'edge_index_nnz': int(len(src)), 'saved_relation_operator_orders': list(range(1, order + 1)), 'saved_paths': [str(out_paths[i]) for i in range(1, order + 1)], 'relation_operator_nnz': {str(k): int(v) for k, v in relation_operator_nnz.items()}, 'clique_count_order_ge2': {str(k): int(v) for k, v in clique_count.items()}, 'edge_stats': edge_stats})
    summary = {'split_root': str(split_root_path), 'node_feature_path': str(node_feature_path_obj), 'num_drug': int(num_drug), 'num_target': int(num_target), 'num_nodes': int(num_nodes), 'order': int(order), 'eps': float(eps), 'restart_probability': float(alpha), 'sim_topk': int(sim_topk), 'sim_min': None if sim_min is None else float(sim_min), 'add_self_loops': bool(add_self_loops), 'remove_homo_cliques': bool(remove_homo_cliques), 'overwrite': bool(overwrite), 'output_prefix': output_prefix, 'num_similarity_edges_directed_global': int(len(sim_src)), 'num_folds': int(len(summary_folds)), 'folds': summary_folds}
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

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate train-sim relation operator for each k-fold split under Data/dti_lists/<dataset>/10fold_split.\nCompared with generate_relation_operator_from_kfold.py, this script adds feature-similarity edges,\nremoves homogeneous higher-order cliques (DDD/TTT), and saves as relation_operator_order_*.pt.')
    parser.add_argument('--split_root', type=str, required=True)
    parser.add_argument('--node_feature_path', type=str, required=True)
    parser.add_argument('--order', type=int, default=4, help='Max relation operator order to generate.')
    parser.add_argument('--eps', type=float, default=1e-08, help='restart propagation epsilon.')
    parser.add_argument('--restart_probability', type=float, default=0.15, help='restart propagation alpha.')
    parser.add_argument('--sim_topk', type=int, default=3, help='Top-k similar neighbors per node.')
    parser.add_argument('--sim_min', type=float, default=0.5, help='Optional minimum cosine similarity for sim edges. Default: no threshold.')
    parser.add_argument('--overwrite', type=_str2bool, default=True, help='Overwrite existing relation operator files.')
    parser.add_argument('--add_self_loops', type=_str2bool, default=False, help='Add self-loops before relation operator.')
    parser.add_argument('--remove_homo_cliques', type=_str2bool, default=True, help='Remove homogeneous higher-order cliques (e.g. DDD/TTT).')
    parser.add_argument('--folds', type=str, default=None, help='Optional comma-separated fold ids, e.g. 1,3,5. Default: all folds.')
    parser.add_argument('--output_prefix', type=str, default='relation_operator_order')
    parser.add_argument('--summary_name', type=str, default='relation_operator_summary.json')
    parser.add_argument('--show_progress', type=_str2bool, default=True)
    args = parser.parse_args()
    summary = run(split_root=args.split_root, node_feature_path=args.node_feature_path, order=args.order, eps=args.eps, alpha=args.restart_probability, sim_topk=args.sim_topk, sim_min=args.sim_min, overwrite=args.overwrite, add_self_loops=args.add_self_loops, remove_homo_cliques=args.remove_homo_cliques, folds=_parse_fold_list(args.folds), output_prefix=args.output_prefix, summary_name=args.summary_name, show_progress=args.show_progress)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
