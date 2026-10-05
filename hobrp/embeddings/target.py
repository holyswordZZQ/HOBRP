import argparse
import os
from typing import Any, Dict, List, Sequence, Tuple
import pandas as pd
import torch
from tqdm import tqdm
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWYX')

def clean_sequence(seq: Any) -> str:
    if seq is None:
        return ''
    cleaned = str(seq).replace(' ', '').replace('\n', '').replace('\r', '').upper()
    return ''.join((ch for ch in cleaned if ch in VALID_AMINO_ACIDS))

def infer_column(df: pd.DataFrame, provided: str, candidates: Sequence[str], kind: str) -> str:
    if provided:
        if provided not in df.columns:
            raise KeyError(f"Column '{provided}' not found for {kind}. Available: {list(df.columns)}")
        return provided
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        hit = normalized.get(c.lower())
        if hit is not None:
            return hit
    raise KeyError(f'Cannot infer {kind} column from {list(df.columns)}. Please set --{kind}_col.')

def load_model(model_name: str, device: torch.device):
    import esm
    print(f'Loading model: {model_name}')
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    return (model, alphabet.get_batch_converter())

def auto_get_repr_layer(model_name: str) -> int:
    for depth in (48, 36, 33, 30, 12, 6):
        if f't{depth}' in model_name:
            return depth
    raise ValueError(f'Cannot infer repr layer from model_name={model_name}.')

def make_batches(data: List[Tuple[str, str]], batch_size: int) -> List[List[Tuple[str, str]]]:
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

@torch.no_grad()
def extract_batch_embeddings(batch_data: List[Tuple[str, str]], model, batch_converter, device: torch.device, repr_layer: int) -> List[Dict[str, Any]]:
    _, _, tokens = batch_converter(batch_data)
    tokens = tokens.to(device)
    token_representations = model(tokens, repr_layers=[repr_layer], return_contacts=False)['representations'][repr_layer]
    outputs: List[Dict[str, Any]] = []
    for i, (target_id, seq) in enumerate(batch_data):
        seq_len = len(seq)
        residue_embedding = token_representations[i, 1:seq_len + 1].detach().cpu()
        if residue_embedding.shape[0] != seq_len:
            raise ValueError(f'Length mismatch for target_id={target_id}: seq_len={seq_len}, embed_len={residue_embedding.shape[0]}')
        outputs.append({'target_id': target_id, 'sequence': seq, 'residue_embedding': residue_embedding})
    return outputs

def parse_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {dtype_name}')

def build_padded_tensor(outputs: List[Dict[str, Any]], max_residues: int, pad_value: float, dtype: torch.dtype) -> Dict[str, Any]:
    if not outputs:
        raise ValueError('No valid outputs to pack.')
    n_targets = len(outputs)
    original_lengths = torch.tensor([item['residue_embedding'].shape[0] for item in outputs], dtype=torch.long)
    k_dim = int(outputs[0]['residue_embedding'].shape[1])
    m_dim = int(original_lengths.max().item()) if max_residues <= 0 else int(max_residues)
    if m_dim <= 0:
        raise ValueError('Resolved residue dimension M must be > 0.')
    embeddings = torch.full((n_targets, m_dim, k_dim), fill_value=pad_value, dtype=dtype)
    mask = torch.zeros((n_targets, m_dim), dtype=torch.bool)
    used_lengths = torch.zeros((n_targets,), dtype=torch.long)
    for i, item in enumerate(outputs):
        residue = item['residue_embedding'].to(dtype=dtype)
        valid_len = min(residue.shape[0], m_dim)
        if valid_len > 0:
            embeddings[i, :valid_len, :] = residue[:valid_len]
            mask[i, :valid_len] = True
            used_lengths[i] = valid_len
    denom = used_lengths.clamp(min=1).unsqueeze(1).to(embeddings.dtype)
    valid_mask = mask.unsqueeze(-1).to(embeddings.dtype)
    mean_embedding = (embeddings * valid_mask).sum(dim=1) / denom
    return {'embedding_3d': embeddings, 'mask': mask, 'lengths': used_lengths, 'original_lengths': original_lengths, 'mean_embedding': mean_embedding, 'target_ids': [item['target_id'] for item in outputs], 'sequences': [item['sequence'] for item in outputs]}

def main():
    parser = argparse.ArgumentParser(description='Extract ESM residue embeddings and save as padded [N, M, K].')
    parser.add_argument('--input_csv', required=True, help='Input target CSV path.')
    parser.add_argument('--output_pt', required=True, help='Output .pt path.')
    parser.add_argument('--id_col', type=str, default='', help='Target id column name. Empty means auto infer.')
    parser.add_argument('--seq_col', type=str, default='', help='Sequence column name. Empty means auto infer.')
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D', help='ESM model name.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda', help="'cuda' or 'cpu'.")
    parser.add_argument('--min_len', type=int, default=1, help='Minimum valid sequence length.')
    parser.add_argument('--max_residues', type=int, default=0, help='M dimension for [N, M, K]. 0 means use longest sequence in input.')
    parser.add_argument('--pad_value', type=float, default=0.0, help='Padding value for [N, M, K].')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float32', help='Output dtype.')
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be > 0')
    if args.min_len <= 0:
        raise ValueError('--min_len must be > 0')
    if args.max_residues < 0:
        raise ValueError('--max_residues must be >= 0')
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    out_dtype = parse_dtype(args.dtype)
    print(f'Using device: {device}')
    print(f'Output dtype: {out_dtype}')
    out_dir = os.path.dirname(args.output_pt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    id_col = infer_column(df, args.id_col, ['target_id', 'targetid', 'protein_id', 'id'], 'id')
    seq_col = infer_column(df, args.seq_col, ['sequence', 'fastas', 'fasta', 'targetsequence', 'protein_fastas', 'target_fasta'], 'seq')
    print(f'Using columns: id_col={id_col}, seq_col={seq_col}')
    records: List[Tuple[str, str]] = []
    dropped = 0
    for _, row in df.iterrows():
        target_id = str(row[id_col])
        seq = clean_sequence(row[seq_col])
        if len(seq) < args.min_len:
            dropped += 1
            continue
        records.append((target_id, seq))
    if not records:
        raise ValueError('No valid sequences after cleaning/filtering.')
    print(f'Valid sequences: {len(records)}')
    print(f'Dropped sequences: {dropped}')
    model, batch_converter = load_model(args.model_name, device)
    repr_layer = auto_get_repr_layer(args.model_name)
    print(f'Using repr layer: {repr_layer}')
    all_outputs: List[Dict[str, Any]] = []
    for batch in tqdm(make_batches(records, args.batch_size), desc='Extracting ESM embeddings'):
        all_outputs.extend(extract_batch_embeddings(batch_data=batch, model=model, batch_converter=batch_converter, device=device, repr_layer=repr_layer))
    packed = build_padded_tensor(outputs=all_outputs, max_residues=args.max_residues, pad_value=args.pad_value, dtype=out_dtype)
    print(f"Embedding shape: {packed['embedding_3d'].shape}")
    torch.save(packed, args.output_pt)
    print(f'Saved to: {args.output_pt}')
    print(f"Shapes -> embedding_3d={tuple(packed['embedding_3d'].shape)}, mask={tuple(packed['mask'].shape)}, mean_embedding={tuple(packed['mean_embedding'].shape)}")
    truncated = int((packed['original_lengths'] > packed['lengths']).sum().item())
    if truncated > 0:
        print(f"Warning: {truncated} sequences were truncated to M={packed['embedding_3d'].shape[1]}")
if __name__ == '__main__':
    main()
