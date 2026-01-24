from __future__ import annotations

from pathlib import Path
from typing import Tuple
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from src.data_loader import load_all_data
from src.utils.git_utils import get_git_commit
from src.utils.provenance import sha256_file, merge_run_metadata

from tqdm.auto import tqdm


def _get_processed_dir() -> Path:
    """
    Ensure data/processed exists under the project root and return its Path.
    """
    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def _get_repo_root() -> Path:
    return Path(".").resolve()


def load_subtask1_for_embeddings() -> pd.DataFrame:
    """
    Load Subtask 1 data for embedding extraction.

    Prefer the feature-augmented parquet file if available:
        data/processed/subtask1_basic_features.parquet
    Otherwise, fall back to raw Subtask 1 from load_all_data().
    """
    processed_path = Path("data/processed/subtask1_basic_features.parquet")

    if processed_path.exists():
        df = pd.read_parquet(processed_path)
        print(f"Loaded Subtask 1 basic features from: {processed_path}")
    else:
        data_bundle = load_all_data()
        df = data_bundle["subtask1"].copy()
        print("Loaded raw Subtask 1 data via load_all_data() (basic features parquet not found).")

    required_cols = {"user_id", "text_id", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Subtask 1 data: {missing}")

    return df


def load_sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return a sentence-transformers encoder.
    """
    print(f"Loading sentence encoder: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def compute_embeddings_for_subtask1(
    df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dense text embeddings for Subtask 1 texts.

    Returns:
        embeddings: np.ndarray of shape (N, D)
        user_ids: np.ndarray of shape (N,)
        text_ids: np.ndarray of shape (N,)
    """
    texts = df["text"].astype(str).tolist()
    user_ids = df["user_id"].to_numpy()
    text_ids = df["text_id"].to_numpy()

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if embeddings.shape[0] != len(texts):
        raise RuntimeError("Embedding count does not match number of texts.")

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, user_ids, text_ids


def _set_determinism(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def compute_embeddings_for_subtask2a_deberta(
    df: pd.DataFrame,
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"user_id", "text_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for subtask2a embeddings: {sorted(missing)}")

    texts = df["text"].fillna("").astype(str).tolist()
    user_ids = df["user_id"].to_numpy()
    text_ids = df["text_id"].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    batch_starts = range(0, len(texts), batch_size)
    for start in tqdm(batch_starts, desc="Subtask2A embeddings", unit="batch"):
        batch_texts = texts[start : start + batch_size]
        with torch.no_grad():
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_emb.astype(np.float32))

    emb = np.concatenate(embeddings, axis=0)
    if emb.shape[0] != len(texts):
        raise RuntimeError("Embedding count does not match number of texts.")
    return emb, user_ids, text_ids


def save_subtask1_embeddings(
    embeddings: np.ndarray,
    user_ids: np.ndarray,
    text_ids: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
) -> Path:
    """
    Save embeddings and identifiers to a compressed .npz file under data/processed/.

    The filename should encode the model name, e.g.:
        subtask1_embeddings_all-MiniLM-L6-v2.npz
    """
    processed_dir = _get_processed_dir()
    safe_model_name = model_name.replace("/", "-")
    out_path = processed_dir / f"subtask1_embeddings_{safe_model_name}.npz"

    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        user_id=user_ids,
        text_id=text_ids,
    )
    print(f"Saved embeddings to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings.")
    parser.add_argument("--task", choices=["subtask1", "subtask2a"], default="subtask1")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick/dev mode (subtask2a only): embeds only the first N rows to validate wiring.",
    )
    args = parser.parse_args()

    repo_root = _get_repo_root()
    _set_determinism(args.seed)

    if args.task == "subtask1":
        df = load_subtask1_for_embeddings()
        model = load_sentence_encoder(model_name=args.model_name)
        embeddings, user_ids, text_ids = compute_embeddings_for_subtask1(
            df, model, batch_size=args.batch_size
        )
        save_subtask1_embeddings(
            embeddings=embeddings,
            user_ids=user_ids,
            text_ids=text_ids,
            model_name=args.model_name.split("/")[-1],
        )
        return

    if not args.run_id:
        raise SystemExit("--run_id is required for subtask2a embeddings.")
    if args.model_name != "microsoft/deberta-v3-base" or args.max_length != 256:
        raise SystemExit(
            "Subtask2a embeddings require --model_name microsoft/deberta-v3-base and --max_length 256."
        )

    data = load_all_data(data_dir=str(repo_root / "data" / "raw"))
    df_raw = data["subtask2a"].copy()
    print(f"Loaded subtask2a rows: {len(df_raw)} from data_dir={repo_root / 'data' / 'raw'}")

    quick_n = 512
    if args.quick:
        df_raw = df_raw.head(quick_n).copy()
        print(f"[QUICK] Truncated subtask2a rows to first {len(df_raw)} (N={quick_n}).")

    embeddings, user_ids, text_ids = compute_embeddings_for_subtask2a_deberta(
        df_raw,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    processed_dir = _get_processed_dir()
    out_path = processed_dir / f"subtask2a_embeddings__{args.run_id}.npz"
    if out_path.exists():
        raise FileExistsError(f"Embeddings file already exists: {out_path}")
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        user_id=user_ids,
        text_id=text_ids,
    )
    npz_sha = sha256_file(out_path)

    merge_run_metadata(
        repo_root=repo_root,
        run_id=args.run_id,
        updates={
            "task": "subtask2a",
            "task_tag": "subtask2a_embeddings",
            "model_name": args.model_name,
            "max_length": args.max_length,
            "pooling": "cls",
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
            "quick": bool(args.quick),
            "quick_n": int(quick_n) if args.quick else None,
            "df_raw_len": int(len(df_raw)),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": get_git_commit(repo_root),
            "embeddings_path": str(out_path),
            "embeddings_sha256": npz_sha,
        },
    )
    print(f"Saved Subtask2A embeddings to: {out_path}")


if __name__ == "__main__":
    main()

