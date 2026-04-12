# SemEval 2026 Task 2 - Emotion Dynamics

This repository contains our SemEval 2026 Task 2 systems for emotion dynamics forecasting across three subtasks:

- **Subtask 1**: Text-level valence and arousal prediction
- **Subtask 2A**: User-level state-change forecasting
- **Subtask 2B**: User-level disposition-change forecasting

The project is organized as one main README (this file) plus dedicated subtask READMEs with full technical and reproduction details.

## Documentation Structure

- `README.md` (main entry point): project overview, setup, and quick navigation
- `README_subtask1.md`: Subtask 1 method and reproduction
- `README_subtask2a.md`: Subtask 2A method and reproduction
- `README_subtask2b.md`: Subtask 2B method and reproduction

If you are new to the repository, start here and then jump to the subtask README you want to run.

## Repository Layout

Key directories and files:

- `src/`: training, prediction, blending, submission, and utility scripts
- `data/`: local data staging (`raw/`, `processed/`) during experiments
- `models/`: local model checkpoints written by training/inference workflows
- `reports/`: split files, predictions, run metadata, and submission artifacts
- `requirements.txt`: pinned Python dependencies used in experiments/reproduction

Note: many generated artifacts are intentionally git-ignored (for size and portability), so some paths are created during local runs.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Use the command patterns below to run each subtask at a high level.

### Subtask 1 (text-level)

Generate submission predictions from a trained run:

```bash
python -m src.submit_subtask1 \
  --run_id <RUN_ID> \
  --checkpoint_dir models/subtask1_transformer/runs/<RUN_ID> \
  --test_path data/raw/test/test_subtask1.csv \
  --max_length 256 \
  --batch_size 8 \
  --scale_arousal_to_valence_range
```

### Subtask 2A (state-change forecasting)

Typical workflow:

1) predict baseline and sequence-model forecasts
2) blend both forecasts
3) write final submission CSV

```bash
# baseline/model forecast generation
python -m src.predict_subtask2a --run_id <RUN_ID> ...

# blending
python -m src.blend_subtask2a_preds \
  --baseline_run_id <BASELINE_RUN_ID> \
  --model_run_id <MODEL_RUN_ID> \
  --alpha_valence 0.40 \
  --alpha_arousal 0.10 \
  --blend_run_id <BLEND_RUN_ID>

# submission writer
python -m src.submit_subtask2a \
  --run_id <BLEND_RUN_ID> \
  --forecast_preds_path reports/preds/subtask2a_forecast_user_preds__<BLEND_RUN_ID>.parquet
```

### Subtask 2B (disposition-change forecasting)

Generate forecasts and write a submission:

```bash
python -m src.predict_subtask2b_user \
  --run_id <RUN_ID> \
  --mode forecast \
  --marker_path data/raw/test/subtask2b_forecasting_user_marker.csv \
  --emb_path <EMBEDDINGS_PATH>

python -m src.submit_subtask2b \
  --run_id <RUN_ID> \
  --pred_path reports/preds/subtask2b_forecast_user_preds__<RUN_ID>.parquet \
  --out_path reports/submissions/subtask2b_submission__<RUN_ID>.csv
```

## Reproducibility Notes

- Use only official SemEval data for final runs.
- Keep split files fixed (seeded unseen-user splits) to compare runs consistently.
- Use the specific artifact paths in each subtask README for reproducible inference.
- Expect minor numeric variation across hardware/software stacks.

## Detailed Subtask Guides

For exact model design, training objectives, hyperparameters, and full reproduction:

- [Subtask 1 guide](README_subtask1.md)
- [Subtask 2A guide](README_subtask2a.md)
- [Subtask 2B guide](README_subtask2b.md)

## Suggested Workflow For New Contributors

1. Set up environment from `requirements.txt`
2. Read this `README.md`
3. Open the subtask README you need
4. Run the quick reproduction path first (inference-only)
5. Run full training only if you need to retrain or extend models

## License / Usage

This repository is intended for research and competition experimentation. Confirm SemEval task rules and data usage terms before redistribution of artifacts.