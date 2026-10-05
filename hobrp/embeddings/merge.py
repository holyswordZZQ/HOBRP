import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

def _to_float32_2d(name: str, value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be 2D, got shape={arr.shape}')
    return arr.astype(np.float32, copy=False)

def _maybe_to_int_ids(ids: np.ndarray) -> np.ndarray:
    if ids.dtype.kind in {'i', 'u'}:
        return ids.astype(np.int64, copy=False)
    try:
        return np.asarray([int(x) for x in ids], dtype=np.int64)
    except (TypeError, ValueError):
        return ids.astype(str)

def _load_drug(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    if 'drug_features' not in data:
        raise KeyError(f'`drug_features` not found in {npz_path}')
    drug_features = _to_float32_2d('drug_features', data['drug_features'])
    if 'drug_ids' in data:
        drug_ids = _maybe_to_int_ids(np.asarray(data['drug_ids']))
    else:
        drug_ids = np.arange(drug_features.shape[0], dtype=np.int64)
    if len(drug_ids) != drug_features.shape[0]:
        raise ValueError(f'drug_ids length mismatch in {npz_path}: ids={len(drug_ids)}, rows={drug_features.shape[0]}')
    return (drug_features, drug_ids)

def _safe_torch_load(pt_path: Path):
    try:
        return torch.load(pt_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(pt_path, map_location='cpu')

def _load_target(pt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = _safe_torch_load(pt_path)
    target_features = None
    target_ids = None
    if isinstance(obj, dict):
        for key in ('embedding', 'target_features', 'mean_embedding'):
            if key in obj:
                target_features = obj[key]
                break
        target_ids = obj.get('target_ids', None)
    elif isinstance(obj, torch.Tensor):
        target_features = obj
    else:
        target_features = obj
    if target_features is None:
        raise KeyError(f'Cannot find target embedding in {pt_path}. Expected one of: embedding/target_features/mean_embedding')
    target_features_np = _to_float32_2d('target_features', target_features)
    if target_ids is None:
        target_ids_np = np.arange(target_features_np.shape[0], dtype=np.int64)
    else:
        target_ids_np = _maybe_to_int_ids(np.asarray(target_ids))
    if len(target_ids_np) != target_features_np.shape[0]:
        raise ValueError(f'target_ids length mismatch in {pt_path}: ids={len(target_ids_np)}, rows={target_features_np.shape[0]}')
    return (target_features_np, target_ids_np)

def _dataset_prefix(path: Path, marker: str) -> str:
    name = path.name
    if marker not in name:
        raise ValueError(f"File name does not contain '{marker}': {path}")
    return name.split(marker, 1)[0]

def _find_pairs(input_dir: Path) -> List[Tuple[str, Path, Path]]:
    drugs: Dict[str, Tuple[str, Path]] = {}
    targets: Dict[str, Tuple[str, Path]] = {}
    for path in sorted(input_dir.glob('*_drug_*.npz')):
        prefix = _dataset_prefix(path, '_drug_')
        key = prefix.lower()
        if key in drugs:
            raise ValueError(f"Duplicate drug file for dataset '{prefix}': {drugs[key][1]} and {path}")
        drugs[key] = (prefix, path)
    for path in sorted(input_dir.glob('*_target_*.pt')):
        prefix = _dataset_prefix(path, '_target_')
        key = prefix.lower()
        if key in targets:
            raise ValueError(f"Duplicate target file for dataset '{prefix}': {targets[key][1]} and {path}")
        targets[key] = (prefix, path)
    common_keys = sorted(set(drugs.keys()) & set(targets.keys()))
    if not common_keys:
        raise FileNotFoundError(f"No paired '*_drug_*.npz' and '*_target_*.pt' found under {input_dir}")
    pairs = []
    for key in common_keys:
        dataset_name = drugs[key][0]
        pairs.append((dataset_name, drugs[key][1], targets[key][1]))
    return pairs

def _resolve_output_dataset_name(dataset_name: str, output_dir: Path) -> str:
    existing = {p.stem.lower(): p.stem for p in output_dir.glob('*.npz')}
    return existing.get(dataset_name.lower(), dataset_name)

def merge_one(dataset_name: str, drug_npz: Path, target_pt: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and (not overwrite):
        print(f'[skip] {output_path} exists. Use --overwrite to replace.')
        return
    drug_features, drug_ids = _load_drug(drug_npz)
    target_features, target_ids = _load_target(target_pt)
    if drug_features.shape[1] != target_features.shape[1]:
        raise ValueError(f'Feature dim mismatch for {dataset_name}: drug_dim={drug_features.shape[1]}, target_dim={target_features.shape[1]}')
    node_features = np.concatenate([drug_features, target_features], axis=0).astype(np.float32, copy=False)
    output_dim = np.asarray([node_features.shape[1]], dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, drug_features=drug_features.astype(np.float32, copy=False), target_features=target_features.astype(np.float32, copy=False), drug_ids=drug_ids, target_ids=target_ids, node_features=node_features, output_dim=output_dim)
    print(f'[ok] {dataset_name} -> {output_path.name} | drug={drug_features.shape}, target={target_features.shape}, node={node_features.shape}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Merge per-dataset drug(.npz) + target(.pt) embeddings into Data/node_features/*.npz format.')
    parser.add_argument('--input_dir', type=str, required=True, help="Directory that contains '*_drug_*.npz' and '*_target_*.pt'.")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write merged node feature npz files.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output npz files.')
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f'input_dir not found: {input_dir}')
    pairs = _find_pairs(input_dir)
    print(f'Found {len(pairs)} dataset pair(s) in {input_dir}')
    for dataset_name, drug_npz, target_pt in pairs:
        out_name = _resolve_output_dataset_name(dataset_name, output_dir)
        out_path = output_dir / f'{out_name}.npz'
        merge_one(dataset_name=dataset_name, drug_npz=drug_npz, target_pt=target_pt, output_path=out_path, overwrite=args.overwrite)
    print('Done.')
if __name__ == '__main__':
    main()
