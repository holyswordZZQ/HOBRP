import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from .model import build_model

def set_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value: true/false')

def _load_csv_edge_list(csv_path: Path) -> List[Tuple[int, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing edge csv: {csv_path}')
    df = pd.read_csv(csv_path)
    if 'drug_idx' not in df.columns or 'target_idx' not in df.columns:
        raise ValueError(f'CSV must contain drug_idx/target_idx: {csv_path}')
    if df.empty:
        return []
    pairs = df[['drug_idx', 'target_idx']].drop_duplicates()
    return [tuple(x) for x in pairs.astype(np.int64).to_numpy().tolist()]

def _infer_num_drug_target_from_fold(train_pos_edges: List[Tuple[int, int]], test_pos_edges: List[Tuple[int, int]]) -> Tuple[int, int]:
    all_edges = train_pos_edges + test_pos_edges
    if not all_edges:
        raise ValueError('No positive edges in train/test csv, cannot infer num_drug/num_target.')
    arr = np.asarray(all_edges, dtype=np.int64)
    return (int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1)

def load_fold_meta(fold_dir: Path) -> Dict:
    fold_dir = fold_dir.resolve()
    split_root = fold_dir.parent
    root_meta = split_root / 'meta.json'
    fold_meta = fold_dir / 'meta.json'
    meta = {}
    if root_meta.exists():
        with open(root_meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    elif fold_meta.exists():
        with open(fold_meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    return meta

def load_node_features(node_feature_path: str, num_drug: int, num_target: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(node_feature_path, allow_pickle=True)
    expected_nodes = int(num_drug + num_target)
    if 'node_features' in data:
        node_features = data['node_features'].astype(np.float32)
        if node_features.shape[0] != expected_nodes:
            raise ValueError(f'node_features rows mismatch: got {node_features.shape[0]}, expected {expected_nodes}')
    elif 'drug_features' in data and 'target_features' in data:
        drug_features = data['drug_features'].astype(np.float32)
        target_features = data['target_features'].astype(np.float32)
        if drug_features.shape[0] != num_drug or target_features.shape[0] != num_target:
            raise ValueError(f'drug_features/target_features rows mismatch with fold meta: drug {drug_features.shape[0]} vs {num_drug}, target {target_features.shape[0]} vs {num_target}')
        node_features = np.concatenate([drug_features, target_features], axis=0).astype(np.float32)
    else:
        raise ValueError(f'Cannot load node features from {node_feature_path}. Need key `node_features` or both `drug_features` and `target_features`.')
    if 'drug_features' in data:
        drug_features = data['drug_features'].astype(np.float32)
        if drug_features.shape[0] != num_drug:
            raise ValueError(f'drug_features rows mismatch: got {drug_features.shape[0]}, expected {num_drug}')
    else:
        drug_features = node_features[:num_drug]
    return (node_features, drug_features)

def build_drug_dissimilarity_neighbors_from_features(drug_features, topk=10):
    num_drug = int(drug_features.shape[0])
    if num_drug <= 1:
        return np.zeros((num_drug, 0), dtype=np.int64)
    topk = int(max(1, min(topk, num_drug - 1)))
    normalized = l2_normalize_rows(drug_features.astype(np.float32))
    sim = np.matmul(normalized, normalized.T).astype(np.float32)
    np.fill_diagonal(sim, np.inf)
    candidate_cols = np.argpartition(sim, kth=topk - 1, axis=1)[:, :topk]
    candidate_scores = np.take_along_axis(sim, candidate_cols, axis=1)
    order = np.argsort(candidate_scores, axis=1)
    return np.take_along_axis(candidate_cols, order, axis=1).astype(np.int64)

def negativeative_mask_from_positive_mask(pos_mask, drug_dissimmat):
    neg_mask = np.zeros_like(pos_mask, dtype=bool)
    pos_drug_idx, pos_target_idx = np.where(pos_mask)
    for d, t in zip(pos_drug_idx.tolist(), pos_target_idx.tolist()):
        neg_mask[drug_dissimmat[d], t] = True
    return neg_mask

def make_single_fold_neg_edges_dissimilarity_style(train_pos_edges: List[Tuple[int, int]], test_pos_edges: List[Tuple[int, int]], num_drug: int, num_target: int, drug_dissimmat: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    train_arr = np.asarray(train_pos_edges, dtype=np.int64)
    test_arr = np.asarray(test_pos_edges, dtype=np.int64)
    all_pos_arr = np.concatenate([train_arr, test_arr], axis=0)
    all_pos_mask = np.zeros((num_drug, num_target), dtype=bool)
    all_pos_mask[all_pos_arr[:, 0], all_pos_arr[:, 1]] = True
    unknown_mask = ~all_pos_mask
    train_pos_mask = np.zeros_like(all_pos_mask, dtype=bool)
    test_pos_mask = np.zeros_like(all_pos_mask, dtype=bool)
    train_pos_mask[train_arr[:, 0], train_arr[:, 1]] = True
    test_pos_mask[test_arr[:, 0], test_arr[:, 1]] = True
    train_neg_mask_candidate = negativeative_mask_from_positive_mask(train_pos_mask, drug_dissimmat)
    train_neg_mask = train_neg_mask_candidate & unknown_mask
    test_neg_mask_candidate = negativeative_mask_from_positive_mask(test_pos_mask, drug_dissimmat)
    test_neg_mask = test_neg_mask_candidate & unknown_mask & ~train_neg_mask
    train_neg_drug, train_neg_target = np.where(train_neg_mask)
    test_neg_drug, test_neg_target = np.where(test_neg_mask)
    train_neg_edges = list(zip(train_neg_drug.tolist(), train_neg_target.tolist()))
    test_neg_edges = list(zip(test_neg_drug.tolist(), test_neg_target.tolist()))
    return (train_neg_edges, test_neg_edges)

def l2_normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return features / norms

def load_relation_operator_by_order(fold_dir: Path, order: int, device: str) -> Dict[int, torch.Tensor]:
    relation_operator = {}
    for i in range(1, int(order) + 1):
        relation_operator_path = fold_dir / f'relation_operator_order_{i}.pt'
        if not relation_operator_path.exists():
            raise FileNotFoundError(f'Missing operator file for order {i}: {relation_operator_path}. Please generate operator first.')
        tensor = torch.load(relation_operator_path, map_location='cpu')
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'operator must be torch.Tensor: {relation_operator_path}, got {type(tensor)}')
        if tensor.layout == torch.sparse_coo:
            tensor = tensor.coalesce()
        relation_operator[i] = tensor.to(device)
    return relation_operator

def compute_binary_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float('nan')
    try:
        aupr = average_precision_score(y_true, probs)
    except ValueError:
        aupr = float('nan')
    return {'acc': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred, zero_division=0), 'precision': precision_score(y_true, y_pred, zero_division=0), 'sen': sensitivity, 'mcc': matthews_corrcoef(y_true, y_pred), 'auc': auc, 'aupr': aupr, 'recall': recall, 'y_true': y_true, 'y_pred': y_pred, 'y_score': probs, 'roc_curve': roc_curve(y_true, probs), 'pr_curve': precision_recall_curve(y_true, probs), 'confusion_matrix': (tn, fp, fn, tp)}

def iterate_edge_batches(edge_pairs, labels, batch_size, shuffle=True):
    num_samples = len(edge_pairs)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_edges = [edge_pairs[i] for i in batch_indices]
        batch_labels = labels[batch_indices]
        yield (batch_edges, batch_labels)

def evaluate_model(model, data_for_model, eval_edges, eval_labels, num_drug, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(data_for_model, eval_edges, num_drug, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = eval_labels.cpu().numpy()
    return compute_binary_metrics(y_true=y_true, probs=probs, threshold=threshold)

def train_one_fold(fold_dir: Path, node_feature_path: str, order: int, negative_topk: int, neg_node_feature_path: Optional[str]=None, add_self_loops: bool=True, gcn_normalize: bool=True, device: str='cuda', seed: int=7777, hidden_dim: int=128, dropout: float=0.2, epochs: int=100, lr: float=0.001, weight_decay: float=0.0001, batch_size: Optional[int]=256, shuffle: bool=True, threshold: float=0.5, selection_metric: str='aupr', K: int=10, alpha: float=0.1, dprate: float=0.0, verbose: bool=True):
    if selection_metric not in {'auc', 'aupr'}:
        raise ValueError("selection_metric must be one of: 'auc', 'aupr'")
    model_name = 'hobrp'
    set_seed(seed)
    fold_dir = fold_dir.resolve()
    train_csv = fold_dir / 'train_pos_edges.csv'
    test_csv = fold_dir / 'test_pos_edges.csv'
    train_pos_edges = _load_csv_edge_list(train_csv)
    test_pos_edges = _load_csv_edge_list(test_csv)
    if not train_pos_edges or not test_pos_edges:
        raise ValueError(f'train/test positive edges cannot be empty: {fold_dir}')
    meta = load_fold_meta(fold_dir)
    if 'num_drug' in meta and 'num_target' in meta:
        num_drug = int(meta['num_drug'])
        num_target = int(meta['num_target'])
    else:
        num_drug, num_target = _infer_num_drug_target_from_fold(train_pos_edges, test_pos_edges)
    num_nodes = int(num_drug + num_target)
    node_features_np, drug_features_np = load_node_features(node_feature_path=node_feature_path, num_drug=num_drug, num_target=num_target)
    if node_features_np.shape[0] != num_nodes:
        raise ValueError(f'node_features rows ({node_features_np.shape[0]}) != num_nodes ({num_nodes}).')
    resolved_neg_node_feature_path = str(Path(neg_node_feature_path).resolve()) if neg_node_feature_path else str(Path(node_feature_path).resolve())
    if Path(resolved_neg_node_feature_path) == Path(node_feature_path).resolve():
        neg_drug_features_np = drug_features_np
    else:
        _, neg_drug_features_np = load_node_features(node_feature_path=resolved_neg_node_feature_path, num_drug=num_drug, num_target=num_target)
    drug_dissimmat = build_drug_dissimilarity_neighbors_from_features(drug_features=neg_drug_features_np, topk=negative_topk)
    train_neg_edges, test_neg_edges = make_single_fold_neg_edges_dissimilarity_style(train_pos_edges=train_pos_edges, test_pos_edges=test_pos_edges, num_drug=num_drug, num_target=num_target, drug_dissimmat=drug_dissimmat)
    if not train_neg_edges or not test_neg_edges:
        raise ValueError('dissimilarity-based negative sampling produced empty train/test negatives. Try larger --negative_topk.')
    train_edges = train_pos_edges + train_neg_edges
    test_edges = test_pos_edges + test_neg_edges
    train_labels = torch.tensor([1] * len(train_pos_edges) + [0] * len(train_neg_edges), dtype=torch.float32, device=device)
    test_labels = torch.tensor([1] * len(test_pos_edges) + [0] * len(test_neg_edges), dtype=torch.float32, device=device)
    node_features_tensor = torch.tensor(node_features_np, dtype=torch.float32)
    emb_dim = int(node_features_tensor.shape[1])
    model = build_model(num_nodes=num_nodes, node_features=node_features_tensor, emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout, order=order, K=K, alpha=alpha, dprate=dprate).to(device)
    operator_container = load_relation_operator_by_order(fold_dir=fold_dir, order=order, device=device)
    data_for_model = SimpleNamespace(x=model.node_features.to(device), operator=operator_container)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_metric_value = -1.0
    best_epoch = -1
    best_state_dict = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= len(train_edges):
            optimizer.zero_grad()
            logits = model(data_for_model, train_edges, num_drug, device)
            loss = criterion(logits, train_labels)
            loss.backward()
            optimizer.step()
        else:
            epoch_loss = 0.0
            num_seen = 0
            for batch_edges, batch_labels in iterate_edge_batches(train_edges, train_labels, batch_size=int(batch_size), shuffle=shuffle):
                optimizer.zero_grad()
                logits = model(data_for_model, batch_edges, num_drug, device)
                batch_loss = criterion(logits, batch_labels)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += float(batch_loss.item()) * len(batch_edges)
                num_seen += len(batch_edges)
            loss = torch.tensor(epoch_loss / max(num_seen, 1), dtype=torch.float32)
        eval_result = evaluate_model(model=model, data_for_model=data_for_model, eval_edges=test_edges, eval_labels=test_labels, num_drug=num_drug, device=device, threshold=threshold)
        current_metric = eval_result[selection_metric]
        if np.isnan(current_metric):
            current_metric = -1
        if current_metric > best_metric_value:
            best_metric_value = float(current_metric)
            best_epoch = int(epoch)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == int(epochs)):
            print(f"Epoch [{epoch:03d}/{epochs}] Loss={loss.item():.4f} AUC={eval_result['auc']:.4f} AUPR={eval_result['aupr']:.4f} F1={eval_result['f1']:.4f} MCC={eval_result['mcc']:.4f}")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_eval = evaluate_model(model=model, data_for_model=data_for_model, eval_edges=test_edges, eval_labels=test_labels, num_drug=num_drug, device=device, threshold=threshold)
    result = {'model_name': model_name, 'fold_dir': str(fold_dir), 'node_feature_path': str(Path(node_feature_path).resolve()), 'neg_node_feature_path': resolved_neg_node_feature_path, 'num_drug': int(num_drug), 'num_target': int(num_target), 'num_nodes': int(num_nodes), 'order': int(order), 'negative_topk': int(negative_topk), 'best_epoch': int(best_epoch), 'best_selection_score': float(best_metric_value), 'selection_metric': selection_metric, 'num_train_pos': int(len(train_pos_edges)), 'num_train_neg': int(len(train_neg_edges)), 'num_test_pos': int(len(test_pos_edges)), 'num_test_neg': int(len(test_neg_edges))}
    result.update(final_eval)
    return (model, result)

def _to_jsonable_result(result: Dict) -> Dict:
    out = {}
    for k, v in result.items():
        if k in {'y_true', 'y_pred', 'y_score'}:
            out[k] = np.asarray(v).tolist()
        elif k in {'roc_curve', 'pr_curve'}:
            out[k] = [np.asarray(x).tolist() for x in v]
        elif k == 'confusion_matrix':
            out[k] = [int(x) for x in v]
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            out[k] = int(v)
        else:
            out[k] = v
    return out
