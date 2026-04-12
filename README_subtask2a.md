# Subtask 2A

Back to main documentation: [README](README.md)

## System Overview
Our Subtask 2A system predicts user-level state-change deltas (Δvalence, Δarousal) for each forecasting user (one row per user). We train only on the official Subtask 2a data using a frozen unseen-user split (seed 42) for reproducibility. Each user is represented by precomputed ModernBERT-base text embeddings and a small set of numeric/history features derived from the user’s recent trajectory. Our primary model is a lightweight sequence regressor over a fixed-length window of recent embeddings (seq_len=32) plus numeric state features, producing a 2D output `[Δvalence_pred, Δarousal_pred]`. For stability, we also fit a simple linear “previous-delta” baseline and blend its forecast with the sequence model to form the final submission.

## Data Used for Training
- Official data only (no external corpora).
- Training/eval split: `reports/splits/subtask2a_unseen_user_seed42.json`.
- Precomputed text embeddings: `data/processed/subtask2a_embeddings__subtask2a_modernbert_base_emb_20260127_214820_seed42.npz`.

## Models
**Primary (sequence model):** A GRU-based sequence regressor that encodes a fixed window of recent embedding vectors (seq_len=32) into a single sequence representation, then fuses it with numeric/history features via a small feed-forward head to predict a 2D output `[Δvalence_pred, Δarousal_pred]`. Checkpoint included at: 

`models/subtask2a_sequence/runs/subtask2a_modernbert_seq_20260127_214800_seed42/model.pt`

**Baseline (linear_prev):** a linear regressor that predicts the next deltas from the previous deltas (used as a baseline and as a stabilizer via blending).

## Model Inputs / Outputs
Input (per user anchor):
- Text embedding sequence `E = [e_{t-31}, …, e_t]` (ModernBERT-base embeddings, fixed window length 32).
- Numeric/history features at time `t`: recent state-change history (previous deltas), time-gap since the prior entry, trajectory summary stats (running mean/EMA, history length), current/previous valence–arousal state when available, and missingness indicators.

Output:
- A 2D prediction `[Δvalence_pred, Δarousal_pred]`.

## How the Model is Trained (Loss, Optimization, Hyperparameters)
- Training entrypoint: `python -m src.train_subtask2a_sequence`
- Optimizer: AdamW (`lr=5e-4`, `weight_decay=0.01`)
- Loss: weighted Huber (`delta=0.5`):  
  `loss = 2.0 * huber(Δvalence) + 1.0 * huber(Δarousal)`
- Regularization: dropout=0.2; gradient clipping=1.0; early stopping patience=2
- Best checkpoint selection: highest validation primary score (mean Pearson r over Δvalence/Δarousal)

## Final Submission (Blended Forecast)
We blend the sequence-model forecast with the linear_prev baseline forecast per target:
- `Δvalence_blend = 0.40 * Δvalence_model + 0.60 * Δvalence_baseline`
- `Δarousal_blend = 0.10 * Δarousal_model + 0.90 * Δarousal_baseline`

Blend run id:
`subtask2a_blend_linprev__subtask2a_modernbert_seq_20260127_214800_seed42__av0p40_aa0p10`

## Training Instance Example
Input:
- User embedding window `E = [e_{t-31}, …, e_t]`
- Numeric/history features `x_num(t)`

Target output:
- `y = [Δvalence_true(u,t), Δarousal_true(u,t)]`

Model prediction and loss:
- `ŷ = SequenceStateRegressor(E, x_num(t))`
- `loss = 2.0 * huber(ŷ_valence - y_valence, δ=0.5) + 1.0 * huber(ŷ_arousal - y_arousal, δ=0.5)`

## Inference Instance Example
Input:
- Forecasting user `u` from the official test cutoff file `data/raw/test/test_subtask2.csv`
- Build the same embedding window + numeric/history features as in training

Outputs:
- Sequence model: `[Δvalence_model, Δarousal_model]`
- Baseline: `[Δvalence_base, Δarousal_base]`
- Final blended forecast:  
  `Δvalence_pred = 0.40*model + 0.60*baseline`  
  `Δarousal_pred = 0.10*model + 0.90*baseline`

Submission row written:
- `user_id, pred_state_change_valence, pred_state_change_arousal` (one row per forecasting user)

## Repro Submission

The ZIP includes the required model artifacts and reference scripts. The code depends on additional modules in the repository. Please clone the repo and run from the repo root to reproduce the submission.
### 0. Clone the repository

Clone the repo and run all commands from the repo root.

### 1. Place required artifacts (from the provided ZIP)

After cloning, copy the following files into the repo at these exact paths (create folders if needed):

- Trained checkpoint (required)
`models/subtask2a_sequence/runs/subtask2a_modernbert_seq_20260127_214800_seed42/model.pt`

- Frozen split (required)
`reports/splits/subtask2a_unseen_user_seed42.json`

- Precomputed embeddings (required)
`data/processed/subtask2a_embeddings__subtask2a_modernbert_base_emb_20260127_214820_seed42.npz`

All other code and data files are already present in the repository.

### 2. Expected repository data layout

These paths should exist in the cloned repo:

- `data/raw/test/test_subtask2.csv` (official test cutoff file used for forecasting)
- `data/raw/train/` (official SemEval training data files used by src.data_loader)

### 3. Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### A) Quick reproduction (recommended: inference-only)
Generate the final submission CSV using the provided embeddings + trained checkpoint.

Inputs:
- Split: `reports/splits/subtask2a_unseen_user_seed42.json`
- Embeddings: `data/processed/subtask2a_embeddings__subtask2a_modernbert_base_emb_20260127_214820_seed42.npz`
- Test cutoffs: `data/raw/test/test_subtask2.csv`
- Checkpoint: `models/subtask2a_sequence/runs/subtask2a_modernbert_seq_20260127_214800_seed42/model.pt`

1) Generate FORECAST parquets (baseline + model)
```bash
# baseline forecast (linear_prev)
python -m src.predict_subtask2a \
  --seed 42 \
  --run_id <BASELINE_RUN_ID> \
  --split_path reports/splits/subtask2a_unseen_user_seed42.json \
  --embeddings_path data/processed/subtask2a_embeddings__subtask2a_modernbert_base_emb_20260127_214820_seed42.npz \
  --forecast_cutoff_path data/raw/test/test_subtask2.csv \
  --write_baseline 1 \
  --write_forecast 1

# sequence model forecast (uses provided checkpoint)
python -m src.predict_subtask2a \
  --seed 42 \
  --run_id <MODEL_RUN_ID> \
  --embeddings_path data/processed/subtask2a_embeddings__subtask2a_modernbert_base_emb_20260127_214820_seed42.npz \
  --model_path models/subtask2a_sequence/runs/subtask2a_modernbert_seq_20260127_214800_seed42/model.pt \
  --forecast_cutoff_path data/raw/test/test_subtask2.csv \
  --write_forecast 1
```

This creates:
`reports/preds/subtask2a_forecast_user_preds__<BASELINE_RUN_ID>.parquet`

`reports/preds/subtask2a_forecast_user_preds__<MODEL_RUN_ID>.parquet`

2) Blend the forecast parquets
```bash
python -m src.blend_subtask2a_preds \
  --baseline_run_id <BASELINE_RUN_ID> \
  --model_run_id <MODEL_RUN_ID> \
  --alpha_valence 0.40 \
  --alpha_arousal 0.10 \
  --blend_run_id <BLEND_RUN_ID>
```

This creates:
`reports/preds/subtask2a_forecast_user_preds__<BLEND_RUN_ID>.parquet`

3) Write the submission CSV
```bash
python -m src.submit_subtask2a \
  --run_id <BLEND_RUN_ID> \
  --forecast_preds_path reports/preds/subtask2a_forecast_user_preds__<BLEND_RUN_ID>.parquet
```

This creates:
`reports/submissions/subtask2a_submission__<BLEND_RUN_ID>.csv`

### B) Full reproduction (optional, includes training)
If you want to retrain the sequence model from scratch (official data only), run training first, then follow Section A (but point --model_path to your newly trained checkpoint).

```bash
python -m src.train_subtask2a_sequence \
  --seed 42 \
  --run_id <MODEL_RUN_ID> \
  --split_path reports/splits/subtask2a_unseen_user_seed42.json \
  --embeddings_path data/processed/subtask2a_embeddings__subtask2a_modernbert_base_emb_20260127_214820_seed42.npz \
  --seq_len 32 \
  --k_state 8 \
  --epochs 10 \
  --lr 0.0005 \
  --weight_decay 0.01 \
  --dropout 0.2 \
  --grad_clip 1.0 \
  --patience 2 \
  --valence_loss huber \
  --huber_delta 0.5 \
  --valence_loss_weight 2.0 \
  --arousal_loss_weight 1.0
```

Then in Section A, use:
`--model_path models/subtask2a_sequence/runs/<MODEL_RUN_ID>/model.pt`. Cloning the repository is required for this option as well.


