"""Dense embeddings via sentence-transformers, cached to disk per
(model, chunk_size, overlap) so ablation sweeps don't re-embed unchanged
chunk sets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from taglish_rag.config import data_path, load_yaml

EMBED_CACHE_DIR = data_path("index", "embeddings")

_model_cache: dict[str, object] = {}


def get_model_config(name: str) -> dict:
    cfg = load_yaml("embeddings.yaml")
    if name not in cfg["models"]:
        raise ValueError(f"Unknown embedding model '{name}'. Options: {list(cfg['models'])}")
    return cfg["models"][name]


def _load_sentence_transformer(hf_id: str):
    from sentence_transformers import SentenceTransformer

    if hf_id not in _model_cache:
        _model_cache[hf_id] = SentenceTransformer(hf_id)
    return _model_cache[hf_id]


def _cache_key(model_name: str, chunk_ids: list[str], is_query: bool) -> str:
    h = hashlib.sha256("|".join(chunk_ids).encode()).hexdigest()[:16]
    kind = "query" if is_query else "passage"
    return f"{model_name}__{kind}__{h}.npy"


def embed_texts(
    model_name: str,
    ids: list[str],
    texts: list[str],
    is_query: bool = False,
    batch_size: int = 32,
) -> np.ndarray:
    """Returns an (n, dim) float32 array, L2-normalized if the model config says so.
    Cached to disk keyed by a hash of the id list, so re-running an ablation axis
    that doesn't touch embeddings (e.g. reranker on/off) is instant.
    """
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = get_model_config(model_name)
    cache_path = EMBED_CACHE_DIR / _cache_key(model_name, ids, is_query)
    if cache_path.exists():
        return np.load(cache_path)

    model = _load_sentence_transformer(cfg["hf_id"])
    prefix = ""
    if is_query and "query_prefix" in cfg:
        prefix = cfg["query_prefix"]
    elif not is_query and "passage_prefix" in cfg:
        prefix = cfg["passage_prefix"]
    prefixed = [prefix + t for t in texts]

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=cfg.get("normalize", True),
        convert_to_numpy=True,
    ).astype("float32")

    np.save(cache_path, embeddings)
    return embeddings
