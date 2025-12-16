from pathlib import Path

import pandas as pd

NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "poutcome",
]

BINARY_FEATURES = ["default", "housing", "loan"]

TRAINING_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def get_project_root() -> Path:
    current_dir = Path(__file__).resolve().parent

    for parent in current_dir.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Project root not found.")


PROJECT_ROOT = get_project_root()


def get_data_dir() -> Path:
    data_dir = PROJECT_ROOT / "data"

    if data_dir.exists():
        return data_dir

    raise RuntimeError("Data dir not found.")


def get_artifacts_dir() -> Path:
    artifacts_dir = PROJECT_ROOT / "artifacts"

    if artifacts_dir.exists():
        return artifacts_dir

    raise RuntimeError("Artifacts dir not found.")


def get_config_dir() -> Path:
    config_dir = PROJECT_ROOT / "config"

    if config_dir.exists():
        return config_dir

    raise RuntimeError("Config dir not found.")


def encode_binary_features(df: pd.DataFrame, binary_features: list[str]) -> None:
    try:
        for bf in binary_features:
            df[bf] = df[bf].map({"yes": 1, "no": 0})
    except KeyError as e:
        print(f"An unexpected key error occurred during binary encoding: {e}")
