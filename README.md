# SemEval 2026 Task 2 — Emotion Dynamics

Code, baselines, and models for SemEval 2026 Task 2: predicting variation in emotional valence and arousal over time from ecological essays. The current focus covers Subtask 1 (per-entry valence/arousal) and Subtask 2A (state-change ΔV/ΔA), with pipelines for data loading, baselines, embeddings, sequence modeling, and evaluation.

## Setup & Installation

1. Clone the repository:

   ```bash
   git clone <THIS_REPO_URL>
   cd semeval-2-emotion-dynamics
   ```

2. Create and activate a virtual environment (example for Python 3.11):

   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
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
python -m src.run_baselines_with_features
```

This will validate the raw CSVs, print basic stats, build Subtask 1 features if needed, and run Subtask 1 baselines (saving `reports/baseline_comparison.csv`).

## Project Structure

- `src/data_loader.py` — load and validate raw CSVs.
- `src/features/basic_features.py` — build simple numeric/text features for Subtask 1.
- `src/run_baselines_with_features.py` — global mean, TF-IDF, and TF-IDF+features baselines.
- `src/embeddings/extract_embeddings.py` — compute sentence embeddings for Subtask 1.
- `src/run_embedding_regressor_subtask1.py` — ridge regressor on embeddings vs TF-IDF.
- `src/sequence_models/` — datasets, simple LSTM regressor, and trainer for Subtask 2A sequences.
- `src/eval/analysis_tools.py` — evaluate trained Subtask 2A model, save predictions/metrics.
- `notebooks/01_eda.ipynb` — exploratory data analysis.
- `notebooks/02_trajectory_analysis.ipynb` — visualize per-user trajectories from predictions.

## Progress So Far

The repository currently includes:

- Subtask 1 baselines (global mean, TF-IDF, TF-IDF + features).
- Embedding-based regressor for Subtask 1.
- LSTM-based sequence model for Subtask 2A.
- Evaluation utilities and trajectory visualization notebook.

