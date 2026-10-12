"""Load YAML config files from config/ once and share them.

Also loads secrets (like GEMINI_API_KEY) from a .env file in the repo root.
"""
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env")


@lru_cache
def load(name: str) -> dict:
    """load("settings") -> contents of config/settings.yaml"""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def settings() -> dict:
    return load("settings")
