# SemEval 2026 Task 2 — Emotion Dynamics

Code, baselines, and models for SemEval 2026 Task 2: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays. The repository currently focuses on Subtask 1 (per-entry valence/arousal) and Subtask 2A (state-change ΔV/ΔA), with initial baselines for Subtask 2B (disposition change), and provides pipelines for data loading, feature extraction, baselines, embeddings, sequence modeling, and evaluation.

## Setup & Installation

1. Clone the repository:

   ```bash
   git clone <THIS_REPO_URL>
   cd semeval-2-emotion-dynamics
   ```

2. Create and activate a virtual environment (example for Python 3.11):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   You may also need to install:

   - `sentence-transformers`
   - `torch` (choose the appropriate CUDA/CPU build for your system)

## Data Layout

Place the SemEval Task 2 training CSVs under `data/raw/`:

- `data/raw/train_subtask1.csv`
- `data/raw/train_subtask2a.csv`
- `data/raw/train_subtask2b.csv`
- `data/raw/train_subtask2b_detailed.csv`
- `data/raw/train_subtask2b_user_disposition_change.csv`

The `src.data_loader` module assumes this layout.

## Quickstart

Once data and dependencies are in place, you can run:

```bash
python -m src.data_loader
python -m src.features.basic_features
python -m src.run_baselines_with_features
```

This will validate the raw CSVs, print basic stats, build Subtask 1 features (`data/processed/subtask1_basic_features.parquet`), and run Subtask 1 baselines (saving `reports/baseline_comparison.csv`).

## Verification Gate

CPU-safe verification scaffold. It performs no training, creates no splits, and does not overwrite artifacts.
Phase VG-0 only validates the contract; later phases add deeper checks.

```bash
python -m src.verify.gate --mode smoke --tasks all --seed 42
python -m src.verify.gate --mode strict --tasks all --seed 42
```

## Project Structure

- `src/data_loader.py` — load and validate raw CSVs.
- `src/features/basic_features.py` — build simple numeric/text features for Subtask 1.
- `src/run_baselines_with_features.py` — global mean, TF-IDF, and TF-IDF+features baselines.
- `src/embeddings/extract_embeddings.py` — compute sentence embeddings for Subtask 1.
- `src/run_embedding_regressor_subtask1.py` — ridge regressor on embeddings vs TF-IDF.
- `src/run_subtask1_model_comparison.py` — k-fold comparison across Subtask 1 models (global mean, TF-IDF, TF-IDF+features, embeddings, embeddings+RandomForest).
- `src/sequence_models/` — datasets, simple LSTM regressor, and trainer for Subtask 2A sequences.
- `src/sequence_models/tune_subtask2a_sequence.py` — small sweep comparing LSTM and Transformer sequence models for Subtask 2A.
- `src/run_subtask2b_baselines.py` — user-level disposition-change baselines for Subtask 2B.
- `src/eval/analysis_tools.py` — evaluate trained Subtask 2A model, save predictions/metrics.
- `notebooks/01_eda.ipynb` — exploratory data analysis.
- `notebooks/02_trajectory_analysis.ipynb` — visualize per-user trajectories from predictions.

## Progress So Far

The repository currently includes:

- **Subtask 1 (per-entry valence/arousal)**  
  - Data loading and normalization.  
  - Baselines: global mean, TF-IDF + Ridge, TF-IDF + simple numeric features + Ridge.  
  - Embedding-based regressor on `all-MiniLM-L6-v2` sentence embeddings.  
  - Cross-validated model comparison across TF-IDF, feature-augmented, and embedding-based models (`run_subtask1_model_comparison.py` → `reports/subtask1_model_comparison.csv`).

- **Subtask 2A (state-change ΔV/ΔA)**  
  - Sequence dataset built from precomputed embeddings.  
  - LSTM-based sequence regressor and basic training script.  
  - Tuning script that compares LSTM and Transformer-based sequence models across a small hyperparameter grid (`tune_subtask2a_sequence.py` → `reports/subtask2a_sequence_model_comparison.csv`).  
  - Evaluation and trajectory visualization tools.

- **Subtask 2B (disposition change)**  
  - User-level view of disposition change built from detailed and user-level tables.  
  - Baseline models: global mean, Ridge on user-level numeric features, RandomForest on the same features (`run_subtask2b_baselines.py` → `reports/subtask2b_baseline_comparison.csv`).

## Additional Scripts

Once embeddings and basic baselines are in place, you can also run:

```bash
# Subtask 1 model comparison (cross-validation across several models)
python -m src.run_subtask1_model_comparison

# Subtask 2A sequence model tuning (LSTM vs Transformer configs)
python -m src.sequence_models.tune_subtask2a_sequence

# Subtask 2B disposition change baselines
python -m src.run_subtask2b_baselines
```

## Phase D (Subtask 2B user MLP)

```bash
python -m src.train_subtask2b_user --seed 42 --run_id subtask2b_phaseD_seed42 \
  --split_path reports/splits/subtask2b_user_disposition_change_unseen_user_seed42.json \
  --emb_path data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz --pooling mean_last

python -m src.predict_subtask2b_user --seed 42 --run_id subtask2b_phaseD_seed42 --mode val \
  --split_path reports/splits/subtask2b_user_disposition_change_unseen_user_seed42.json \
  --emb_path data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz

python -m src.eval.phase0_eval --task subtask2b --seed 42 --run_id subtask2b_phaseD_seed42
python -m src.verify.subtask2b --seed 42 --run_id subtask2b_phaseD_seed42
python -m src.predict_subtask2b_user --seed 42 --run_id subtask2b_phaseD_seed42 --mode forecast \
  --marker_path data/raw/test/subtask2b_forecasting_user_marker.csv \
  --emb_path data/processed/subtask2b_embeddings__deberta-v3-base__ml256.npz
python -m src.submit_subtask2b --run_id subtask2b_phaseD_seed42
```

## Phase 1 (Frozen-Split Transformer)

```bash
# CPU smoke run (tiny bounded run)
python -m src.train_subtask1_transformer \
  --split_path reports/splits/subtask1_unseen_user_seed42.json \
  --quick

# Colab/GPU training (full)
python -m src.train_subtask1_transformer \
  --split_path reports/splits/subtask1_unseen_user_seed42.json \
  --model_name microsoft/deberta-v3-base \
  --epochs 3 \
  --batch_size 8

# Prediction (writes per-row + per-user parquet)
python -m src.predict_subtask1_transformer \
  --split_path reports/splits/subtask1_unseen_user_seed42.json \
  --ckpt_dir models/subtask1_transformer/best

# Model comparison including transformer
python -m src.run_subtask1_model_comparison \
  --include_transformer \
  --split_path reports/splits/subtask1_unseen_user_seed42.json

# Ensemble (3 seeds; average preds across checkpoints)
python -m src.predict_subtask1_transformer \
  --split_path reports/splits/subtask1_unseen_user_seed42.json \
  --ensemble_ckpt_dirs models/subtask1_transformer/best_seed42,models/subtask1_transformer/best_seed43,models/subtask1_transformer/best_seed44
```

### Run-Targeted Eval (Subtask 1)
```bash
python -m src.eval.phase0_eval --task subtask1 --model_tag subtask1_transformer --run_id <RUN_ID>
python -m src.eval.phase0_eval --task subtask1 --model_tag subtask1_transformer --pred_path reports/preds/subtask1_val_preds__<RUN_ID>.parquet
python -m src.run_subtask1_model_comparison --include_transformer --run_id <RUN_ID>
```

## Run-ID’d artifacts (contract)
- run_id format: `subtask1_transformer_20260121_154233_seed42`
- never overwrite by default
- predictions: `reports/preds/` with `__{run_id}` suffix
- per-run metadata: `reports/runs/{run_id}.json`
- future checkpoints: `models/subtask1_transformer/runs/{run_id}/`
- task is canonical: `subtask1|subtask2a|subtask2b`
- task_tag is richer and drives run_id generation (e.g., `subtask1_transformer`)
- metadata timestamp includes timezone offset
- This phase adds contracts + utilities only; training/predict/eval scripts will be upgraded in a later phase.

