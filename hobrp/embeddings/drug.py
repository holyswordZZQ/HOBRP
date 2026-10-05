import argparse
import os
from typing import Any, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

def infer_column(df: pd.DataFrame, provided: str, candidates: Sequence[str], kind: str) -> str:
    if provided:
        if provided not in df.columns:
            raise KeyError(f"Column '{provided}' not found for {kind}. Available: {list(df.columns)}")
        return provided
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        hit = normalized.get(candidate.lower())
        if hit is not None:
            return hit
    raise KeyError(f'Cannot infer {kind} column from {list(df.columns)}. Please set --{kind}_col.')

def first_non_empty(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value) and str(value).strip() != '':
            return str(value).strip()
    return ''

def load_drug_csv(input_csv: str, id_col: str='', smiles_col: str='') -> Tuple[pd.DataFrame, bool]:
    df = pd.read_csv(input_csv)
    smiles_column = infer_column(df, smiles_col, ['smiles', 'SMILES'], 'smiles')
    inferred_id_col = id_col
    if not inferred_id_col:
        normalized = {str(col).strip().lower(): col for col in df.columns}
        for name in ('drugid', 'drug_id', 'id'):
            if name in normalized:
                inferred_id_col = normalized[name]
                break
    smiles = df[smiles_column].fillna('').astype(str).str.strip()
    if inferred_id_col:
        raw_ids = df[inferred_id_col]
        numeric_ids = pd.to_numeric(raw_ids, errors='coerce')
        id_is_numeric = bool(numeric_ids.notna().all())
        if id_is_numeric:
            drug_ids = numeric_ids.astype(np.int64)
        else:
            drug_ids = raw_ids.fillna('').astype(str).str.strip()
    else:
        id_is_numeric = True
        drug_ids = np.arange(len(df), dtype=np.int64)
    table = pd.DataFrame({'drug_id': drug_ids, 'smiles': smiles})
    if id_is_numeric:
        table = table.groupby('drug_id', as_index=False)['smiles'].agg(first_non_empty).sort_values('drug_id').reset_index(drop=True)
    else:
        table = table.groupby('drug_id', sort=False, as_index=False)['smiles'].agg(first_non_empty).reset_index(drop=True)
    table['smiles'] = table['smiles'].fillna('').astype(str).str.strip()
    table = table[table['smiles'] != ''].reset_index(drop=True)
    if table.empty:
        raise ValueError(f'No valid SMILES found in {input_csv}')
    return (table, id_is_numeric)

def extract_cls_embeddings(records: List[Tuple[Any, str]], model_name: str, batch_size: int) -> Tuple[List[Any], List[str], np.ndarray, int]:
    try:
        from unimol_tools import UniMolRepr
    except ImportError as exc:
        raise ImportError('unimol_tools is required for drug embedding extraction. Please install it in the current Python environment.') from exc
    if batch_size <= 0:
        raise ValueError('--batch_size must be > 0')
    repr_model = UniMolRepr(data_type='molecule', model_name=model_name, batch_size=batch_size)
    kept_ids: List[Any] = []
    kept_smiles: List[str] = []
    vectors: List[np.ndarray] = []
    failed = 0
    error_samples: List[str] = []

    def to_2d_float32(x: Any) -> np.ndarray:
        if hasattr(x, 'detach') and hasattr(x, 'cpu'):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f'Embedding tensor must be 2D or 1D, got shape={arr.shape}')
        return arr

    def parse_cls_repr(reprs: Any) -> np.ndarray:
        if isinstance(reprs, dict):
            preferred = ['cls_repr', 'molecule_repr', 'mol_repr', 'embeddings', 'repr']
            for key in preferred:
                if key in reprs:
                    return to_2d_float32(reprs[key])
            for key, value in reprs.items():
                key_low = str(key).lower()
                if 'repr' in key_low and ('cls' in key_low or 'mol' in key_low):
                    return to_2d_float32(value)
            raise KeyError(f'No CLS representation key found. Available keys: {list(reprs.keys())}')
        return to_2d_float32(reprs)
    try:
        all_smiles = [item[1] for item in records]
        reprs = repr_model.get_repr(all_smiles, return_atomic_reprs=False, return_tensor=True)
        all_vecs = parse_cls_repr(reprs)
        if all_vecs.ndim == 2 and all_vecs.shape[0] == len(records):
            kept_ids = [item[0] for item in records]
            kept_smiles = all_smiles
            return (kept_ids, kept_smiles, all_vecs.astype(np.float32), 0)
        raise ValueError(f'Unexpected full-batch shape={all_vecs.shape}, expected rows={len(records)}')
    except Exception as exc:
        if len(error_samples) < 5:
            error_samples.append(f'full_batch: {type(exc).__name__}: {exc}')
    for start in tqdm(range(0, len(records), batch_size), desc='Extracting UniMol CLS embeddings'):
        chunk = records[start:start + batch_size]
        chunk_smiles = [item[1] for item in chunk]
        try:
            reprs = repr_model.get_repr(chunk_smiles, return_atomic_reprs=False, return_tensor=True)
            chunk_vecs = parse_cls_repr(reprs)
            if chunk_vecs.ndim != 2 or chunk_vecs.shape[0] != len(chunk):
                raise ValueError(f'Unexpected cls_repr shape={chunk_vecs.shape} for batch size={len(chunk)}')
            kept_ids.extend((item[0] for item in chunk))
            kept_smiles.extend(chunk_smiles)
            vectors.append(chunk_vecs)
            continue
        except Exception as exc:
            if len(error_samples) < 5:
                error_samples.append(f'batch_start={start}: {type(exc).__name__}: {exc}')
        for drug_id, smi in chunk:
            try:
                reprs = repr_model.get_repr([smi], return_atomic_reprs=False, return_tensor=True)
                one_vec = parse_cls_repr(reprs)
                if one_vec.ndim != 2 or one_vec.shape[0] != 1:
                    raise ValueError(f'Unexpected cls_repr shape for single input: {one_vec.shape}')
                kept_ids.append(drug_id)
                kept_smiles.append(smi)
                vectors.append(one_vec)
            except Exception as exc:
                failed += 1
                if len(error_samples) < 5:
                    error_samples.append(f'drug_id={drug_id}: {type(exc).__name__}: {exc}')
    if not vectors:
        extra = '\n'.join(error_samples) if error_samples else 'No additional error details captured.'
        raise ValueError(f'Failed to extract any drug embeddings from UniMol.\n{extra}')
    if error_samples:
        print('UniMol extraction had partial failures (showing up to 5):')
        for msg in error_samples:
            print(f'  - {msg}')
    embeddings = np.concatenate(vectors, axis=0).astype(np.float32)
    return (kept_ids, kept_smiles, embeddings, failed)

def reduce_with_pca_and_pad(embeddings: np.ndarray, target_dim: int, random_state: int) -> Tuple[np.ndarray, int]:
    if embeddings.ndim != 2:
        raise ValueError(f'embeddings should be 2D, got shape={embeddings.shape}')
    if target_dim <= 0:
        raise ValueError('--pca_dim must be > 0')
    n_samples, in_dim = embeddings.shape
    used_dim = min(target_dim, n_samples, in_dim)
    if used_dim <= 0:
        return (np.zeros((n_samples, target_dim), dtype=np.float32), 0)
    pca = PCA(n_components=used_dim, random_state=random_state)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    if used_dim < target_dim:
        pad = np.zeros((n_samples, target_dim - used_dim), dtype=np.float32)
        reduced = np.concatenate([reduced, pad], axis=1)
    return (reduced, used_dim)

def main() -> None:
    parser = argparse.ArgumentParser(description='Read drug.csv, extract UniMol CLS embeddings from SMILES, and save PCA-128 drug features.')
    parser.add_argument('--input_csv', type=str, required=True, help='Input drug csv path (e.g. Data/dti_lists/BindingDB/drug.csv).')
    parser.add_argument('--output_npz', type=str, required=True, help='Output npz path.')
    parser.add_argument('--id_col', type=str, default='', help='Drug id column name. Empty means auto infer.')
    parser.add_argument('--smiles_col', type=str, default='', help='SMILES column name. Empty means auto infer.')
    parser.add_argument('--model_name', type=str, default='unimolv2', help='UniMol model name.')
    parser.add_argument('--batch_size', type=int, default=64, help='UniMol embedding batch size.')
    parser.add_argument('--pca_dim', type=int, default=128, help='Target PCA dimension.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for PCA.')
    parser.add_argument('--max_samples', type=int, default=0, help='Only keep first N drugs for quick debugging.')
    args = parser.parse_args()
    table, id_is_numeric = load_drug_csv(args.input_csv, id_col=args.id_col, smiles_col=args.smiles_col)
    if args.max_samples > 0:
        table = table.head(args.max_samples).copy()
    records = list(zip(table['drug_id'].tolist(), table['smiles'].tolist()))
    kept_ids, kept_smiles, raw_embeddings, failed = extract_cls_embeddings(records=records, model_name=args.model_name, batch_size=args.batch_size)
    reduced, used_dim = reduce_with_pca_and_pad(embeddings=raw_embeddings, target_dim=args.pca_dim, random_state=args.seed)
    out_dir = os.path.dirname(args.output_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if id_is_numeric:
        drug_ids = np.asarray(kept_ids, dtype=np.int64)
        id_dtype = 'int64'
    else:
        drug_ids = np.asarray(kept_ids, dtype=str)
        id_dtype = 'str'
    np.savez_compressed(args.output_npz, drug_ids=drug_ids, smiles=np.asarray(kept_smiles, dtype=str), drug_features=reduced.astype(np.float32), raw_drug_features=raw_embeddings.astype(np.float32), output_dim=np.asarray([args.pca_dim], dtype=np.int64), pca_used_dim=np.asarray([used_dim], dtype=np.int64), raw_dim=np.asarray([raw_embeddings.shape[1]], dtype=np.int64), failed_smiles=np.asarray([failed], dtype=np.int64), drug_id_dtype=np.asarray([id_dtype]))
    print(f'Input CSV                : {args.input_csv}')
    print(f'Total valid input drugs  : {len(records)}')
    print(f'Encoded drugs            : {len(kept_ids)}')
    print(f'Failed SMILES            : {failed}')
    print(f'Raw embedding shape      : {raw_embeddings.shape}')
    print(f'PCA feature shape        : {reduced.shape}')
    print(f'PCA used dimension       : {used_dim}')
    print(f'Saved to                 : {args.output_npz}')
if __name__ == '__main__':
    main()
