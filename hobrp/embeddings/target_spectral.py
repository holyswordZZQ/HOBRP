import os
import math
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.fftpack import dct, idct

def safe_entropy(x: np.ndarray, eps: float=1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s < eps:
        return 0.0
    p = x / s
    p = np.maximum(p, eps)
    return float(-(p * np.log(p)).sum())

def stable_rank_from_singulars(s: np.ndarray, eps: float=1e-12) -> float:
    if s.size == 0:
        return 0.0
    s2 = s ** 2
    denom = np.max(s2)
    if denom < eps:
        return 0.0
    return float(np.sum(s2) / denom)

def effective_rank_from_singulars(s: np.ndarray, eps: float=1e-12) -> float:
    if s.size == 0:
        return 0.0
    s = np.maximum(s, 0.0)
    total = s.sum()
    if total < eps:
        return 0.0
    p = s / total
    p = np.maximum(p, eps)
    h = -(p * np.log(p)).sum()
    return float(np.exp(h))

def energy_ratio_topk(s: np.ndarray, k: int) -> float:
    if s.size == 0:
        return 0.0
    total = np.sum(s ** 2)
    if total <= 0:
        return 0.0
    k = min(k, s.size)
    return float(np.sum(s[:k] ** 2) / total)

def condition_number_from_singulars(s: np.ndarray, eps: float=1e-12) -> float:
    if s.size == 0:
        return 0.0
    s = np.asarray(s)
    s_pos = s[s > eps]
    if s_pos.size == 0:
        return 0.0
    return float(np.max(s_pos) / np.min(s_pos))

def compute_band_boundaries(length: int, low_ratio: float=0.125, mid_ratio: float=0.5) -> Dict[str, Tuple[int, int]]:
    low_end = max(1, int(math.floor(length * low_ratio)))
    mid_end = max(low_end + 1, int(math.floor(length * mid_ratio)))
    mid_end = min(mid_end, length)
    return {'low': (0, low_end), 'mid': (low_end, mid_end), 'high': (mid_end, length)}

def dct_along_sequence(E: np.ndarray) -> np.ndarray:
    return dct(E, axis=0, norm='ortho')

def idct_along_sequence(E_hat: np.ndarray) -> np.ndarray:
    return idct(E_hat, axis=0, norm='ortho')

def reconstruct_band(E_hat: np.ndarray, start: int, end: int) -> np.ndarray:
    E_band_hat = np.zeros_like(E_hat)
    if end > start:
        E_band_hat[start:end, :] = E_hat[start:end, :]
    return idct_along_sequence(E_band_hat)

def band_basic_stats(E_band: np.ndarray, eps: float=1e-12) -> Dict[str, float]:
    flat = E_band.reshape(-1)
    abs_flat = np.abs(flat)
    fro_norm = float(np.linalg.norm(E_band, ord='fro'))
    mean_abs = float(abs_flat.mean()) if abs_flat.size > 0 else 0.0
    std_val = float(flat.std()) if flat.size > 0 else 0.0
    max_abs = float(abs_flat.max()) if abs_flat.size > 0 else 0.0
    sparsity_l1_l2 = float(abs_flat.sum() / (fro_norm + eps)) if abs_flat.size > 0 else 0.0
    return {'fro_norm': fro_norm, 'mean_abs': mean_abs, 'std': std_val, 'max_abs': max_abs, 'l1_over_l2': sparsity_l1_l2}

def svd_stats(E_band: np.ndarray, topk_singular: int=10, eps: float=1e-12) -> Dict[str, float]:
    try:
        s = np.linalg.svd(E_band, compute_uv=False, full_matrices=False)
    except np.linalg.LinAlgError:
        noise = 1e-06 * np.random.randn(*E_band.shape)
        s = np.linalg.svd(E_band + noise, compute_uv=False, full_matrices=False)
    feats = {}
    padded = np.zeros(topk_singular, dtype=np.float64)
    copy_len = min(topk_singular, s.size)
    padded[:copy_len] = s[:copy_len]
    for i in range(topk_singular):
        feats[f'sv_{i + 1}'] = float(padded[i])
    total_sv = float(np.sum(s)) + eps
    total_energy = float(np.sum(s ** 2)) + eps
    for i in range(topk_singular):
        feats[f'sv_ratio_{i + 1}'] = float(padded[i] / total_sv)
        feats[f'sv_energy_ratio_{i + 1}'] = float(padded[i] ** 2 / total_energy)
    feats['rank_arestart_scoresox'] = float(np.sum(s > eps))
    feats['stable_rank'] = stable_rank_from_singulars(s, eps=eps)
    feats['effective_rank'] = effective_rank_from_singulars(s, eps=eps)
    feats['condition_number'] = condition_number_from_singulars(s, eps=eps)
    feats['sv_entropy'] = safe_entropy(s, eps=eps)
    feats['top1_energy_ratio'] = energy_ratio_topk(s, 1)
    feats['top3_energy_ratio'] = energy_ratio_topk(s, 3)
    feats['top5_energy_ratio'] = energy_ratio_topk(s, 5)
    feats['top10_energy_ratio'] = energy_ratio_topk(s, 10)
    feats['sum_singular'] = float(np.sum(s))
    feats['sum_singular_sq'] = float(np.sum(s ** 2))
    feats['mean_singular'] = float(np.mean(s)) if s.size > 0 else 0.0
    feats['std_singular'] = float(np.std(s)) if s.size > 0 else 0.0
    return feats

def extract_one_target_features(E: np.ndarray, topk_singular: int=10, low_ratio: float=0.125, mid_ratio: float=0.5) -> Tuple[np.ndarray, Dict[str, float]]:
    assert E.ndim == 2, f'E should be 2D, got shape={E.shape}'
    L, D = E.shape
    if L < 4:
        basic = {'length': float(L), 'dim': float(D), 'fallback_mean_abs': float(np.abs(E).mean()), 'fallback_std': float(E.std()), 'fallback_fro_norm': float(np.linalg.norm(E, ord='fro'))}
        vec = np.array(list(basic.values()), dtype=np.float32)
        return (vec, basic)
    boundaries = compute_band_boundaries(L, low_ratio=low_ratio, mid_ratio=mid_ratio)
    E_hat = dct_along_sequence(E)
    all_feats: Dict[str, float] = {'length': float(L), 'dim': float(D)}
    embedding_parts: List[float] = []
    for band_name in ['low', 'mid', 'high']:
        start, end = boundaries[band_name]
        E_band = reconstruct_band(E_hat, start, end)
        basic_stats = band_basic_stats(E_band)
        svd_feat = svd_stats(E_band, topk_singular=topk_singular)
        coeff = E_hat[start:end, :]
        band_coeff_energy = float(np.sum(coeff ** 2))
        total_coeff_energy = float(np.sum(E_hat ** 2)) + 1e-12
        band_coeff_energy_ratio = band_coeff_energy / total_coeff_energy
        band_feats = {f'{band_name}_start': float(start), f'{band_name}_end': float(end), f'{band_name}_width': float(max(0, end - start)), f'{band_name}_coeff_energy': band_coeff_energy, f'{band_name}_coeff_energy_ratio': band_coeff_energy_ratio}
        for k, v in basic_stats.items():
            band_feats[f'{band_name}_{k}'] = float(v)
        for k, v in svd_feat.items():
            band_feats[f'{band_name}_{k}'] = float(v)
        all_feats.update(band_feats)
        for key in sorted(band_feats.keys()):
            embedding_parts.append(float(band_feats[key]))
    vec = np.array(embedding_parts, dtype=np.float32)
    return (vec, all_feats)

def load_input_pt(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location='cpu')
    if not isinstance(obj, dict):
        raise TypeError(f'Expected a dict in {path}, but got {type(obj)}. Please save a dict with keys like embedding_3d and mask.')
    if 'embedding_3d' not in obj:
        raise KeyError("Missing key 'embedding_3d' in input .pt")
    if 'mask' not in obj:
        raise KeyError("Missing key 'mask' in input .pt")
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pt', type=str, required=True, help='输入 ESM 残基 embedding 的 .pt 文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--topk_singular', type=int, default=10, help='每个频带保留前 k 个奇异值统计')
    parser.add_argument('--low_ratio', type=float, default=0.125, help='低频截止比例')
    parser.add_argument('--mid_ratio', type=float, default=0.5, help='中频截止比例')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_input_pt(args.input_pt)
    embedding_3d = data['embedding_3d']
    mask = data['mask']
    target_ids = data.get('target_ids', None)
    if isinstance(embedding_3d, torch.Tensor):
        embedding_3d = embedding_3d.cpu()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu()
    embedding_3d = embedding_3d.numpy() if isinstance(embedding_3d, torch.Tensor) else np.asarray(embedding_3d)
    mask = mask.numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if embedding_3d.ndim != 3:
        raise ValueError(f'embedding_3d should be 3D, got shape={embedding_3d.shape}')
    if mask.ndim != 2:
        raise ValueError(f'mask should be 2D, got shape={mask.shape}')
    N, Lmax, D = embedding_3d.shape
    if mask.shape != (N, Lmax):
        raise ValueError(f'mask shape {mask.shape} does not match embedding_3d shape {embedding_3d.shape}')
    if target_ids is None:
        target_ids = [f'target_{i}' for i in range(N)]
    else:
        target_ids = list(target_ids)
        if len(target_ids) != N:
            raise ValueError(f'len(target_ids)={len(target_ids)} but N={N}')
    embedding_list = []
    rows = []
    for i in tqdm(range(N), desc='Building frequency-SVD target embeddings'):
        tid = str(target_ids[i])
        valid_len = int(mask[i].sum())
        if valid_len <= 0:
            raise ValueError(f'Target {tid} has valid_len <= 0')
        E = embedding_3d[i, :valid_len, :].astype(np.float64)
        vec, feat_dict = extract_one_target_features(E, topk_singular=args.topk_singular, low_ratio=args.low_ratio, mid_ratio=args.mid_ratio)
        embedding_list.append(vec)
        row = {'target_id': tid}
        row.update(feat_dict)
        rows.append(row)
    embeddings = np.stack(embedding_list, axis=0)
    df = pd.DataFrame(rows)
    npy_path = os.path.join(args.output_dir, 'target_freq_svd_embedding.npy')
    pt_path = os.path.join(args.output_dir, 'target_freq_svd_embedding.pt')
    csv_path = os.path.join(args.output_dir, 'target_freq_svd_feature_table.csv')
    np.save(npy_path, embeddings)
    torch.save({'target_ids': target_ids, 'embedding': torch.from_numpy(embeddings), 'feature_names': [c for c in df.columns if c != 'target_id'], 'feature_table': df, 'config': {'topk_singular': args.topk_singular, 'low_ratio': args.low_ratio, 'mid_ratio': args.mid_ratio}}, pt_path)
    df.to_csv(csv_path, index=False)
    print(f'Saved npy: {npy_path}')
    print(f'Saved pt : {pt_path}')
    print(f'Saved csv: {csv_path}')
    print(f'Final embedding shape: {embeddings.shape}')
if __name__ == '__main__':
    main()
