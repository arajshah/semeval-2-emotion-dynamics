# Subtask 2B

Back to main documentation: [README](README.md)

## System Overview
Our Subtask 2B system predicts delta disposition-change valence and delta disposition-change arousal at the user level. We train only on the official Subtask 2b data using a frozen unseen-user split (seed 42) to keep results comparable and reproducible. Each user is represented by precomputed Transformer text embeddings (DeBERTa-v3-base, max length 256) aggregated into a single user vector by combining the mean embedding over the user's history with the embedding of the user's most recent text, plus a small set of numeric history summary features that are z-normalized using train-only statistics saved in `models/subtask2b_user/runs/subtask2b_phaseD_seed42/norm_stats.json` (feature construction code: `src/subtask2b_features.py`). The predictive model is a feed-forward MLP that maps the concatenated feature vector to a 2D output `[delta_valence, delta_arousal]`. Training uses AdamW and a weighted MSE objective, selecting the best checkpoint by the validation primary score (mean Pearson r over the two targets).

## Data Used for Training
- Official data only (no external corpora).
- User-level targets: loaded via `load_all_data()["subtask2b_user"]` (user disposition-change labels).
- Text-level histories: loaded via `load_all_data()["subtask2b"]` (per-user text records used to build features).
- Frozen split used for train/val: `reports/splits/subtask2b_user_disposition_change_unseen_user_seed42.json`.
- Precomputed embeddings used: `data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz`.

## Model
Feed-forward MLP regressor (3-layer hidden stack with LayerNorm + GELU + Dropout) operating on a concatenated feature vector consisting of (i) pooled DeBERTa-v3-base text embeddings and (ii) normalized numeric history features (normalization fit on train only and stored in `norm_stats.json`). Checkpointed at:
`models/subtask2b_user/runs/subtask2b_phaseD_seed42/{model.pt, config.json, norm_stats.json}`

## Model Inputs / Outputs
Input (per user):
- Aggregated text embedding vector (from the user's history, pooled as `mean_last`).
- Numeric history summary features (normalized using train-fitted stats saved in `norm_stats.json`).
- These are concatenated into one vector of size `input_dim` (stored in `models/subtask2b_user/runs/subtask2b_phaseD_seed42/config.json`).

Output (per user):
- A 2D vector `[pred_dispo_change_valence, pred_dispo_change_arousal]` representing predicted delta changes.

## How the Model is Trained (Loss, Optimization, Hyperparameters)
- Training entrypoint: `python -m src.train_subtask2b_user`
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- Loss: `MSE(valence) + lambda_arousal * MSE(arousal)` with `lambda_arousal=1.0`
- Batch size: 256; epochs: 30; seed: 42; pooling: `mean_last`
- Best checkpoint selection: highest validation primary score (mean Pearson r across valence/arousal)

## Training Instance Example
Input:
- User text history embeddings: e1,...,ek from `data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz`
- Aggregate embedding: `e_user = mean_last_pool(e1..ek)`
- Numeric summary features: `x_num = normalize(history_stats(u); norm_stats fitted on train only)`
- Final input vector: `x = concat(e_user, x_num)`

Target output:
- `y = [disposition_change_valence(u), disposition_change_arousal(u)]` from official user-level labels

Training step:
- `y_hat = MLP(x)`
- `loss = MSE(y_hat[0], y[0]) + 1.0 * MSE(y_hat[1], y[1])`

## Inference Instance Example
Input:
- User u listed as forecasting in: `data/raw/test/subtask2b_forecasting_user_marker.csv`
- Build features exactly as in training:
  `x = concat(mean_last_pool(embeddings(u)), normalize(history_stats(u)))`
- Load checkpoint:
  `models/subtask2b_user/runs/subtask2b_phaseD_seed42/`

Predicted output:
- `y_hat = [pred_dispo_change_valence(u), pred_dispo_change_arousal(u)]`

Submission row written:
- `user_id, pred_dispo_change_valence, pred_dispo_change_arousal` (one row per forecasting user)

## Repro Submission

The ZIP includes the required model artifacts and reference scripts. The code depends on additional modules in the repository. Please clone the repo and run from the repo root to reproduce the submission.

### 0. Clone the repository
Clone the repo and run all commands from the repo root.

### 1. Place required artifacts (from the provided ZIP)
After cloning, copy the following files into the repo at these exact paths (create folders if needed):

- Trained checkpoint (required)  
`models/subtask2b_user/runs/subtask2b_phaseD_seed42/model.pt`

- Run config (required)  
`models/subtask2b_user/runs/subtask2b_phaseD_seed42/config.json`

- Normalization stats (required)  
`models/subtask2b_user/runs/subtask2b_phaseD_seed42/norm_stats.json`

- Frozen split (required for training provenance)  
`reports/splits/subtask2b_user_disposition_change_unseen_user_seed42.json`

- Precomputed embeddings (required for inference)  
`data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz`

All other code and data files are already present in the repository.

### 2. Expected repository data layout
These paths should exist in the cloned repo:

- `data/raw/test/subtask2b_forecasting_user_marker.csv` (official forecasting marker file used for inference)
- `data/raw/train/` (official SemEval training data files used by `src.data_loader`)

### 3. Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### A. Quick reproduction (recommended: inference only)
Generate the final submission CSV using the provided embeddings and checkpoint.

Generate forecast predictions
```bash
python -m src.predict_subtask2b_user \
  --seed 42 \
  --run_id subtask2b_phaseD_seed42 \
  --mode forecast \
  --marker_path data/raw/test/subtask2b_forecasting_user_marker.csv \
  --emb_path data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz \
  --overwrite 1
```

Creates:
`reports/preds/subtask2b_forecast_user_preds__subtask2b_phaseD_seed42.parquet`

Write the submission CSV
```bash
python -m src.submit_subtask2b \
  --run_id subtask2b_phaseD_seed42 \
  --pred_path reports/preds/subtask2b_forecast_user_preds__subtask2b_phaseD_seed42.parquet \
  --out_path reports/submissions/_repro_check__subtask2b_phaseD_seed42.csv \
  --clip_to_theoretical_bounds 1 \
  --overwrite 1
```

Creates:
`reports/submissions/_repro_check__subtask2b_phaseD_seed42.csv`

### B. Full reproduction (optional, includes training)
If you want to retrain the model from scratch, run training first, then run Section A.

```bash
python -m src.train_subtask2b_user \
  --seed 42 \
  --run_id <TRAIN_RUN_ID> \
  --split_path reports/splits/subtask2b_user_disposition_change_unseen_user_seed42.json \
  --emb_path data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz
```

Then run Section A using `--run_id <TRAIN_RUN_ID>` and the checkpoint written under:
`models/subtask2b_user/runs/<TRAIN_RUN_ID>/`

