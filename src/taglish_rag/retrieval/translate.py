"""Optional query translation to English (one ablation axis: does
translating a Tagalog/Taglish query to English before retrieval improve
recall against an English-only corpus?).

Uses a small local MarianMT model (free, no API key, runs on CPU or GPU)
rather than an LLM, so this ablation works without any provider key
configured. Loads the tokenizer/model directly rather than via
transformers.pipeline("translation", ...) -- that convenience wrapper's
task registry has proven brittle across transformers versions (it dropped
the bare "translation" task name at some point), while
AutoTokenizer/AutoModelForSeq2SeqLM are stable, low-level APIs.
"""
from __future__ import annotations

_tokenizer = None
_model = None

MODEL_ID = "Helsinki-NLP/opus-mt-tl-en"


def _get_translator():
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        if torch.cuda.is_available():
            _model = _model.to("cuda")
    return _tokenizer, _model


def translate_to_english(text: str) -> str:
    import torch

    tokenizer, model = _get_translator()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if model.device.type == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)
