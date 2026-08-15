"""Token-based sliding-window chunking (config-driven: size + overlap)."""
from __future__ import annotations

import tiktoken

_encoder_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoder(name: str) -> tiktoken.Encoding:
    if name not in _encoder_cache:
        _encoder_cache[name] = tiktoken.get_encoding(name)
    return _encoder_cache[name]


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    tokenizer: str = "cl100k_base",
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    enc = _get_encoder(tokenizer)
    tokens = enc.encode(text, disallowed_special=())
    if not tokens:
        return []

    stride = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(enc.decode(window))
        if start + chunk_size >= len(tokens):
            break
        start += stride
    return chunks
