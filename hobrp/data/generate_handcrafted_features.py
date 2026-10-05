import argparse
import os
from collections import Counter
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
DIPEPTIDE_INDEX = {dp: idx for idx, dp in enumerate(DIPEPTIDES)}

def read_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    lower = path.lower()
    if lower.endswith('.csv'):
        return pd.read_csv(path)
    if lower.endswith('.xlsx') or lower.endswith('.xls'):
        return pd.read_excel(path)
    raise ValueError(f'Only support .csv / .xlsx / .xls, got: {path}')

def infer_column(df, candidates, file_path):
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(f'None of columns {candidates} found in {file_path}. Available columns: {list(df.columns)}')

def first_non_empty(series):
    for value in series:
        if pd.notna(value) and str(value) != '':
            return str(value)
    return ''

def load_drug_table(drug_path):
    df = read_table(drug_path)
    smiles_col = infer_column(df, ['SMILES', 'smiles'], drug_path)
    id_col = None
    for candidate in ['drug_id', 'drugid']:
        if candidate in {str(col).strip().lower() for col in df.columns}:
            id_col = infer_column(df, [candidate], drug_path)
            break
    if id_col is None:
        output = pd.DataFrame({'drug_id': np.arange(len(df), dtype=np.int64), 'SMILES': df[smiles_col].fillna('').astype(str)})
    else:
        output = pd.DataFrame({'drug_id': pd.to_numeric(df[id_col], errors='raise').astype(np.int64), 'SMILES': df[smiles_col].fillna('').astype(str)})
        output = output.groupby('drug_id', as_index=False)['SMILES'].agg(first_non_empty).sort_values('drug_id').reset_index(drop=True)
    output['drug_id'] = output['drug_id'].astype(np.int64)
    output['SMILES'] = output['SMILES'].fillna('').astype(str)
    return output

def load_target_table(target_path):
    df = read_table(target_path)
    fasta_col = infer_column(df, ['protein_fastas', 'FASTA', 'TargetSequence', 'target_fasta', 'fastas'], target_path)
    id_col = None
    for candidate in ['target_id', 'targetid', 'protein_id']:
        if candidate in {str(col).strip().lower() for col in df.columns}:
            id_col = infer_column(df, [candidate], target_path)
            break
    if id_col is None:
        output = pd.DataFrame({'target_id': np.arange(len(df), dtype=np.int64), 'FASTA': df[fasta_col].fillna('').astype(str)})
    else:
        output = pd.DataFrame({'target_id': pd.to_numeric(df[id_col], errors='raise').astype(np.int64), 'FASTA': df[fasta_col].fillna('').astype(str)})
        output = output.groupby('target_id', as_index=False)['FASTA'].agg(first_non_empty).sort_values('target_id').reset_index(drop=True)
    output['target_id'] = output['target_id'].astype(np.int64)
    output['FASTA'] = output['FASTA'].fillna('').astype(str)
    return output

def resolve_feature_paths(drug_path='', target_path='', dataset='', data_dir='Data', dti_lists_dir='Data/dti_lists'):
    if drug_path and target_path:
        return (drug_path, target_path)
    if drug_path or target_path:
        raise ValueError('Please provide both --drug_path and --target_path together.')
    if not dataset:
        raise ValueError('Provide either both --drug_path/--target_path or --dataset.')
    candidates = [os.path.join(data_dir, dataset), os.path.join(dti_lists_dir, dataset)]
    dataset_dir = None
    for candidate in candidates:
        if os.path.isdir(candidate):
            dataset_dir = candidate
            break
    if dataset_dir is None:
        raise FileNotFoundError(f"Cannot find dataset directory for '{dataset}'. Tried: {candidates}")
    drug_candidates = [os.path.join(dataset_dir, 'drugs.xlsx'), os.path.join(dataset_dir, 'drugs.xls'), os.path.join(dataset_dir, 'drug.csv')]
    target_candidates = [os.path.join(dataset_dir, 'targets.xlsx'), os.path.join(dataset_dir, 'targets.xls'), os.path.join(dataset_dir, 'target.csv')]
    resolved_drug = next((p for p in drug_candidates if os.path.exists(p)), None)
    resolved_target = next((p for p in target_candidates if os.path.exists(p)), None)
    if resolved_drug is None:
        raise FileNotFoundError(f'Cannot find drugs file under {dataset_dir}. Tried: {drug_candidates}')
    if resolved_target is None:
        raise FileNotFoundError(f'Cannot find targets file under {dataset_dir}. Tried: {target_candidates}')
    return (resolved_drug, resolved_target)

def bitvect_to_array(fp, length):
    arr = np.zeros((length,), dtype=np.float32)
    if fp is not None:
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def featurize_smiles(smiles, morgan_bits=1024, morgan_radius=2):
    if not smiles:
        return np.zeros((morgan_bits + 167 + 8,), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((morgan_bits + 167 + 8,), dtype=np.float32)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    morgan_arr = bitvect_to_array(morgan_fp, morgan_bits)
    maccs_arr = bitvect_to_array(maccs_fp, 167)
    descriptor_values = np.asarray([float(Descriptors.MolWt(mol)), float(Descriptors.MolLogP(mol)), float(Descriptors.TPSA(mol)), float(Descriptors.NumHDonors(mol)), float(Descriptors.NumHAcceptors(mol)), float(Descriptors.NumRotatableBonds(mol)), float(Descriptors.RingCount(mol)), float(Descriptors.FractionCSP3(mol))], dtype=np.float32)
    return np.concatenate([morgan_arr, maccs_arr, descriptor_values], axis=0)

def amino_acid_composition(seq):
    seq = seq or ''
    counts = Counter((ch for ch in seq if ch in AMINO_ACIDS))
    total = sum(counts.values())
    if total == 0:
        return np.zeros((len(AMINO_ACIDS),), dtype=np.float32)
    return np.asarray([counts[aa] / total for aa in AMINO_ACIDS], dtype=np.float32)

def dipeptide_composition(seq):
    seq = ''.join((ch for ch in seq or '' if ch in AMINO_ACIDS))
    vec = np.zeros((len(DIPEPTIDES),), dtype=np.float32)
    if len(seq) < 2:
        return vec
    total = len(seq) - 1
    for i in range(total):
        dp = seq[i:i + 2]
        idx = DIPEPTIDE_INDEX.get(dp)
        if idx is not None:
            vec[idx] += 1.0
    if total > 0:
        vec /= float(total)
    return vec

def basic_sequence_physchem(seq):
    seq = ''.join((ch for ch in seq or '' if ch in AMINO_ACIDS))
    length = float(len(seq))
    if length == 0:
        return np.zeros((8,), dtype=np.float32)
    counts = Counter(seq)
    aromatic = counts['F'] + counts['W'] + counts['Y'] + counts['H']
    polar = counts['S'] + counts['T'] + counts['N'] + counts['Q'] + counts['C']
    positive = counts['K'] + counts['R'] + counts['H']
    negative = counts['D'] + counts['E']
    aliphatic = counts['A'] + counts['V'] + counts['I'] + counts['L'] + counts['M']
    gly_pro = counts['G'] + counts['P']
    return np.asarray([length, aromatic / length, polar / length, positive / length, negative / length, aliphatic / length, gly_pro / length, counts['C'] / length], dtype=np.float32)

def build_target_tfidf_svd(sequences, svd_dim, seed):
    if svd_dim <= 0:
        return np.zeros((len(sequences), 0), dtype=np.float32)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), lowercase=False)
    tfidf = vectorizer.fit_transform(sequences)
    max_dim = min(svd_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if max_dim <= 0:
        return np.zeros((len(sequences), 0), dtype=np.float32)
    svd = TruncatedSVD(n_components=max_dim, random_state=seed)
    return svd.fit_transform(tfidf).astype(np.float32)

def safe_project(features, output_dim, seed):
    if features.shape[1] == 0:
        return np.zeros((features.shape[0], output_dim), dtype=np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    max_dim = min(output_dim, scaled.shape[0], scaled.shape[1])
    if max_dim <= 0:
        return np.zeros((features.shape[0], output_dim), dtype=np.float32)
    if max_dim == scaled.shape[1]:
        projected = scaled.astype(np.float32)
    else:
        pca = PCA(n_components=max_dim, random_state=seed)
        projected = pca.fit_transform(scaled).astype(np.float32)
    if projected.shape[1] < output_dim:
        pad = np.zeros((projected.shape[0], output_dim - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected, pad], axis=1)
    return projected

def generate_node_representation(drug_path, target_path, output_path, output_dim=128, seed=42, morgan_bits=1024, morgan_radius=2, target_tfidf_dim=128):
    np.random.seed(seed)
    drug_table = load_drug_table(drug_path)
    target_table = load_target_table(target_path)
    drug_ids = drug_table['drug_id'].to_numpy(dtype=np.int64)
    target_ids = target_table['target_id'].to_numpy(dtype=np.int64)
    drug_smiles = drug_table['SMILES'].tolist()
    target_fastas = target_table['FASTA'].tolist()
    drug_raw = np.stack([featurize_smiles(smiles, morgan_bits=morgan_bits, morgan_radius=morgan_radius) for smiles in drug_smiles], axis=0)
    target_aac = np.stack([amino_acid_composition(seq) for seq in target_fastas], axis=0)
    target_dipep = np.stack([dipeptide_composition(seq) for seq in target_fastas], axis=0)
    target_phys = np.stack([basic_sequence_physchem(seq) for seq in target_fastas], axis=0)
    target_tfidf = build_target_tfidf_svd(target_fastas, svd_dim=target_tfidf_dim, seed=seed)
    target_raw = np.concatenate([target_aac, target_dipep, target_phys, target_tfidf], axis=1)
    drug_projected = safe_project(drug_raw, output_dim=output_dim, seed=seed)
    target_projected = safe_project(target_raw, output_dim=output_dim, seed=seed)
    node_features = np.concatenate([drug_projected, target_projected], axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, drug_features=drug_projected.astype(np.float32), target_features=target_projected.astype(np.float32), drug_ids=drug_ids, target_ids=target_ids, node_features=node_features, output_dim=np.asarray([output_dim], dtype=np.int64))
    print('=' * 60)
    print('Node Representation Generated')
    print(f'Drug file         : {drug_path}')
    print(f'Target file       : {target_path}')
    print(f'Num drugs         : {len(drug_ids)}')
    print(f'Num targets       : {len(target_ids)}')
    print(f'Drug feature shape: {drug_projected.shape}')
    print(f'Target feat shape : {target_projected.shape}')
    print(f'Node feature shape: {node_features.shape}  (drug concat target)')
    print(f'Saved to          : {output_path}')
    print('=' * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_path', required=True)
    parser.add_argument('--target_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--morgan_bits', type=int, default=1024)
    parser.add_argument('--morgan_radius', type=int, default=2)
    parser.add_argument('--target_tfidf_dim', type=int, default=128)
    args = parser.parse_args()
    generate_node_representation(**vars(args))
if __name__ == '__main__':
    main()
