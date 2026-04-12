# Subtask 1

Back to main documentation: [README](README.md)

## System Overview
Our Subtask 1 system predicts valence and arousal for each text entry (one row per text). We train only on the official Subtask 1 data using a frozen unseen user split (seed 43) for reproducibility. The model is a fine tuned ModernBERT base encoder with a lightweight regression head that outputs a 2D prediction [pred_valence, pred_arousal].

During training we use scaled arousal: arousal is represented in the same range as valence [-2, 2] to stabilize optimization. During inference we invert this scaling back to the official arousal range [0, 2] before writing the submission file.


## Data Used for Training
- Official data only (no external corpora).
- Frozen split (unseen user, seed 43):  
  `reports/splits/subtask1_unseen_user_seed43.json`
- Official test file used for submission inference:  
  `data/raw/test/test_subtask1.csv`

## Model
- Encoder: `answerdotai/ModernBERT-base`
- Head: MLP regressor over the pooled [CLS] representation
- Output: 2D float vector `[pred_valence, pred_arousal]`

## Model Inputs / Outputs
Input (per text row):
- text (tokenized up to max_length=256)

Model does not require labels at inference time.

Output (per text row):
- pred_valence in [-2, 2]
- pred_arousal in [0, 2] after inverting the training scale and clipping

## How the Model is Trained (Loss, Optimization, Hyperparameters)
- Training entrypoint: `python -m src.train_subtask1_transformer`
- Optimizer: AdamW
- Loss: Weighted Smooth L1 (Huber) regression  
  - `loss = SmoothL1(pred_valence, true_valence) + lambda_arousal * SmoothL1(pred_arousal, true_arousal)`  
  - If `scale_arousal_to_valence_range=True`, the arousal target is linearly rescaled from `[0, 2]` to `[-2, 2]` during training.
- Key hyperparameters for the best run:
  - max_length=256
  - epochs=6
  - lr=2e-5
  - dropout=0.05
  - batch_size=8
  - grad_accum_steps=2
  - lambda_arousal=3
  - scale_arousal_to_valence_range=True

## Inference and Submission Format
Inference is run on the official test file `data/raw/test/test_subtask1.csv`. The submission CSV contains one row per test entry with:

`user_id, text_id, pred_valence, pred_arousal`

Output path:
`reports/submissions/pred_subtask1__<RUN_ID>.csv`

## Training Instance Example
Input (one training row):
- `user_id`: `u_001`
- `text_id`: `t_0001`
- `text`: `"I finally finished my project and I feel relieved."`

Output (ground-truth labels used for training):
- `valence` = `1.20`
- `arousal` = `0.85`

## Inference Instance Example
Input (one test row):
- `user_id`: `u_999`
- `text_id`: `t_0420`
- `text`: `"I can’t stop worrying about what comes next."`

Output (model prediction written to submission):
- `pred_valence` = `-1.10`
- `pred_arousal` = `1.35`


## Repro Submission
The ZIP includes the required model artifacts and reference scripts. The code depends on additional modules in the repository. Please clone the repo and run from the repo root to reproduce the submission.

### 0. Clone the repository
Clone the repo and run all commands from the repo root.

### 1. Place required artifacts (from the provided ZIP)
After cloning, copy the following files into the repo at these exact paths (create folders if needed).

- Trained checkpoint directory (required)  
Copy the run directory to repo root:  
`models/subtask1_transformer/runs/subtask1_modernbert_base_ml256_lr2e-05_ep6__I2_bs8_do0.05_lam3_scaleA__seed43_20260127_221941/`

- Frozen split (required for training provenance and reproducibility)  
`reports/splits/subtask1_unseen_user_seed43.json`

All other code and data files are already present in the repository.

### 2. Expected repository data layout
These paths should exist in the cloned repo:
- `data/raw/test/test_subtask1.csv` (official test file used for submission inference)
- `data/raw/train/` (official training data files used by src.data_loader)
- `src/` (training, inference, evaluation modules)

### 3. Environment setup
Install the pinned dependencies from `requirements.txt`. We have tested on Linux x86_64 + CUDA (Colab). Other platforms (e.g., macOS arm64 / CPU) may produce small numeric differences in outputs due to low-level kernel/precision differences.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


### A. Quick reproduction (recommended: inference only)
Run the Subtask 1 submission writer (this is the script that mirrors the Colab inference cells).

```bash
python -m src.submit_subtask1 \
  --run_id subtask1_modernbert_base_ml256_lr2e-05_ep6__I2_bs8_do0.05_lam3_scaleA__seed43_20260127_221941 \
  --checkpoint_dir models/subtask1_transformer/runs/subtask1_modernbert_base_ml256_lr2e-05_ep6__I2_bs8_do0.05_lam3_scaleA__seed43_20260127_221941 \
  --test_path data/raw/test/test_subtask1.csv \
  --max_length 256 \
  --batch_size 8 \
  --scale_arousal_to_valence_range
```

Creates:
`reports/submissions/pred_subtask1__subtask1_modernbert_base_ml256_lr2e-05_ep6__I2_bs8_do0.05_lam3_scaleA__seed43_20260127_221941.csv`

### B. Full reproduction (optional, includes training)
Train from scratch (official data only), then run Section A with your new --run_id.

```bash
python -m src.train_subtask1_transformer \
  --model_name answerdotai/ModernBERT-base \
  --max_length 256 \
  --dropout 0.05 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --epochs 6 \
  --seed 43 \
  --regime unseen_user \
  --amp fp16 \
  --grad_checkpointing \
  --head_type simple \
  --split_path reports/splits/subtask1_unseen_user_seed43.json \
  --lambda_arousal 3 \
  --scale_arousal_to_valence_range \
  --run_id <TRAIN_RUN_ID>
```

The trained checkpoint will be written under:
`models/subtask1_transformer/runs/<TRAIN_RUN_ID>/`
