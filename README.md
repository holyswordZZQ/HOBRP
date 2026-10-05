# HOBRP

HOBRP is a higher-order biological relation propagation framework for drug-target interaction prediction. This repository contains the minimal code required to prepare node representations, construct fold-specific heterogeneous graphs, extract higher-order relations, build relation propagation operators, and train the HOBRP link predictor.

## Repository layout

- `hobrp/embeddings/`: Uni-Mol drug embeddings, ESM target embeddings, target spectral compression, and feature merging.
- `hobrp/data/`: fold generation and optional handcrafted node features.
- `hobrp/datasets/*`: drug informations, target informations and positive DTIs of corresponding dataset.
- `hobrp/relations/`: similarity-edge construction, higher-order relation extraction, and sparse relation operator generation.
- `hobrp/model.py`: HOBRP encoder and DTI link prediction head.
- `hobrp/training.py`: fold loading, negative sampling, training, and evaluation.
- `train.py`: multi-fold training entry point.

The repository intentionally excludes analysis scripts, plots, checkpoints, generated operators, cached files, and baseline implementations.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Data convention

Each dataset directory contains `drug.csv`, `target.csv`, and `dti.csv`. Drug and target identifiers must map consistently to zero-based row indices. Fold generation writes `fold_1` through `fold_k`, each containing `train_pos_edges.csv` and `test_pos_edges.csv`.

All input and output paths are mandatory. The code does not assume a project directory, dataset location, or output location.

## Pipeline

Generate folds:

```bash
python -m hobrp.data.split_folds --data_root /path/to/datasets --k 10 --out_name 10fold_split
```

Generate drug embeddings:

```bash
python -m hobrp.embeddings.drug --input_csv /path/to/drug.csv --output_npz /path/to/drug_features.npz
```

Generate and compress target embeddings:

```bash
python -m hobrp.embeddings.target --input_csv /path/to/target.csv --output_pt /path/to/target_residue_embeddings.pt
python -m hobrp.embeddings.target_spectral --input_pt /path/to/target_residue_embeddings.pt --output_dir /path/to/target_features
```

Merge drug and target features:

```bash
python -m hobrp.embeddings.merge --input_dir /path/to/component_features --output_dir /path/to/node_features
```

Construct the heterogeneous graph, extract higher-order relations, and generate fold-specific relation operators:

```bash
python -m hobrp.relations.build_operators --split_root /path/to/10fold_split --node_feature_path /path/to/node_features.npz --order 4 --output_prefix relation_operator_order
```

Train HOBRP:

```bash
python train.py --split_root /path/to/10fold_split --node_feature_path /path/to/node_features.npz --output_dir /path/to/results --folds 1,2,3,4,5,6,7,8,9,10 --order 4 --device cuda
```

Relation operators must be generated independently for every training fold. Test edges are never used to construct graph relations or propagation operators.
