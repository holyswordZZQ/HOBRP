import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
DRUG_COL_CANDS = ['drug_id', 'drugid']
TARGET_COL_CANDS = ['target_id', 'targetid', 'protein_id']
LABEL_COL_CANDS = ['label', 'binding', 'bind', 'y']

def _infer_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]
    return None

def _load_positive_edges_with_maps(csv_path: str) -> Tuple[List[Tuple[int, int]], int, int, str, str, Optional[str]]:
    df = pd.read_csv(csv_path)
    drug_col = _infer_column(df, DRUG_COL_CANDS)
    target_col = _infer_column(df, TARGET_COL_CANDS)
    if drug_col is None or target_col is None:
        raise ValueError(f'Cannot find drug/target columns in {csv_path}. Available columns: {list(df.columns)}')
    label_col = _infer_column(df, LABEL_COL_CANDS)
    if label_col is not None:
        labels = pd.to_numeric(df[label_col], errors='coerce').fillna(0)
        df = df[labels == 1]
    pairs = df[[drug_col, target_col]].drop_duplicates().reset_index(drop=True)
    if pairs.shape[0] == 0:
        raise ValueError(f'No positive pairs after filtering in {csv_path}.')
    drug_ids = sorted(pairs[drug_col].astype(int).unique().tolist())
    target_ids = sorted(pairs[target_col].astype(int).unique().tolist())
    drug_map = {eid: idx for idx, eid in enumerate(drug_ids)}
    target_map = {eid: idx for idx, eid in enumerate(target_ids)}
    edges: List[Tuple[int, int]] = []
    for _, row in pairs.iterrows():
        edges.append((drug_map[int(row[drug_col])], target_map[int(row[target_col])]))
    return (edges, len(drug_ids), len(target_ids), str(drug_col), str(target_col), label_col)

def _save_edge_csv(path: str, edges: List[Tuple[int, int]]) -> None:
    pd.DataFrame(edges, columns=['drug_idx', 'target_idx']).to_csv(path, index=False)

def _list_dataset_dirs(data_root: str) -> List[str]:
    dirs = []
    for name in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, name)
        if os.path.isdir(full):
            dirs.append(full)
    return dirs

def _list_csv_files(dataset_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(dataset_dir)):
        full = os.path.join(dataset_dir, name)
        if os.path.isfile(full) and name.lower().endswith('.csv'):
            files.append(full)
    return files

def _should_skip_root_csv(dataset_dir: str, csv_path: str) -> bool:
    dataset_name = os.path.basename(dataset_dir)
    if dataset_name.lower() == 'dti_lists':
        return True
    return False

def _build_one_csv_splits(csv_path: str, out_dir: str, k: int, seed: int) -> Dict:
    edges, num_drug, num_target, drug_col, target_col, label_col = _load_positive_edges_with_maps(csv_path)
    edge_arr = np.asarray(edges, dtype=np.int64)
    if edge_arr.shape[0] < 2:
        raise ValueError(f'Need at least 2 edges to split, got {edge_arr.shape[0]} in {csv_path}')
    k_used = int(min(int(k), int(edge_arr.shape[0])))
    if k_used < 2:
        raise ValueError(f'Invalid k after clipping for {csv_path}: {k_used}')
    os.makedirs(out_dir, exist_ok=True)
    kf = KFold(n_splits=k_used, shuffle=True, random_state=int(seed))
    fold_meta = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(edge_arr), start=1):
        fold_dir = os.path.join(out_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        train_edges = [tuple(x) for x in edge_arr[train_idx].tolist()]
        test_edges = [tuple(x) for x in edge_arr[test_idx].tolist()]
        _save_edge_csv(os.path.join(fold_dir, 'train_pos_edges.csv'), train_edges)
        _save_edge_csv(os.path.join(fold_dir, 'test_pos_edges.csv'), test_edges)
        fold_meta.append({'fold': int(fold_idx), 'num_train_pos': int(len(train_edges)), 'num_test_pos': int(len(test_edges))})
    meta = {'source_csv': os.path.abspath(csv_path), 'k_requested': int(k), 'k_used': int(k_used), 'seed': int(seed), 'num_pos_edges': int(edge_arr.shape[0]), 'num_drug': int(num_drug), 'num_target': int(num_target), 'drug_col': str(drug_col), 'target_col': str(target_col), 'label_col': None if label_col is None else str(label_col), 'folds': fold_meta}
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta

def run(data_root: str, k: int, seed: int, out_name: str) -> Dict:
    data_root = os.path.normpath(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'data_root not found: {data_root}')
    if int(k) < 2:
        raise ValueError('k must be >= 2')
    all_summaries = []
    dataset_dirs = _list_dataset_dirs(data_root)
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        csv_files = _list_csv_files(dataset_dir)
        if not csv_files:
            continue
        usable_csvs = []
        for csv_path in csv_files:
            if _should_skip_root_csv(dataset_dir, csv_path):
                continue
            try:
                df_cols = list(pd.read_csv(csv_path, nrows=0).columns)
            except Exception:
                continue
            col_set = {str(c).strip().lower() for c in df_cols}
            has_drug = any((c in col_set for c in DRUG_COL_CANDS))
            has_target = any((c in col_set for c in TARGET_COL_CANDS))
            if has_drug and has_target:
                usable_csvs.append(csv_path)
        if not usable_csvs:
            continue
        base_out = os.path.join(dataset_dir, out_name)
        os.makedirs(base_out, exist_ok=True)
        one_file_mode = len(usable_csvs) == 1
        dataset_report = {'dataset': dataset_name, 'outputs': []}
        for csv_path in usable_csvs:
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            out_dir = base_out if one_file_mode else os.path.join(base_out, stem)
            meta = _build_one_csv_splits(csv_path=csv_path, out_dir=out_dir, k=k, seed=seed)
            dataset_report['outputs'].append({'source_csv': os.path.abspath(csv_path), 'output_dir': os.path.abspath(out_dir), 'k_used': int(meta['k_used']), 'num_pos_edges': int(meta['num_pos_edges'])})
        with open(os.path.join(base_out, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_report, f, ensure_ascii=False, indent=2)
        all_summaries.append(dataset_report)
    final_summary = {'data_root': os.path.abspath(data_root), 'k': int(k), 'seed': int(seed), 'out_name': str(out_name), 'datasets': all_summaries}
    with open(os.path.join(data_root, f'{out_name}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    return final_summary

def main() -> None:
    parser = argparse.ArgumentParser(description='Create K-fold CSV splits from Data/dti_lists/*/*.csv.\nFor each dataset, outputs are saved under: Data/dti_lists/<dataset>/kfold_split/')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--k', type=int, default=10, help='Number of folds.')
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--out_name', type=str, default='10fold_split', help='Output folder name under each dataset directory.')
    args = parser.parse_args()
    summary = run(data_root=args.data_root, k=args.k, seed=args.seed, out_name=args.out_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
