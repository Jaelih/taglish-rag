"""YAML config loading with light validation."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

load_dotenv(REPO_ROOT / ".env", override=False)


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def data_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath("data", *parts)


def eval_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath("eval", *parts)


def results_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath("results", *parts)
