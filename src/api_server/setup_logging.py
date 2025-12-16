import logging.config
import yaml
from utils import get_config_dir


def load_logging_config(env: str = "dev") -> dict:
    """
    Loads logging config from file and applies environment-specific changes.
    Returns the resulting dictionary.
    """
    logging_config_path = get_config_dir() / "logging.yaml"

    if not logging_config_path.exists():
        print(f"CRITICAL: Logging configuration file not found at {logging_config_path}")
        return {}

    with open(logging_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if env.lower() == "dev":
        for logger_name in ["app", "uvicorn.access", "uvicorn.error"]:
            if logger_name in config["loggers"]:
                config["loggers"][logger_name]["handlers"] = ["console_handler"]
                config["loggers"][logger_name]["level"] = "DEBUG"

    return config


def setup_logging(config: dict):
    """Applies the dictionary configuration to the Python logging module."""
    if config:
        logging.config.dictConfig(config)
        print("Logging module configured successfully.")