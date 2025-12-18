from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import pathlib
from utils import PROJECT_ROOT


class AppSettings(BaseSettings):
    ENV: str = "dev"  # Mapped to APP_ENV
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 4

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_prefix="APP_",
    )


APP_SETTINGS_INSTANCE = None


def get_settings() -> AppSettings:
    """Retrieves the single, resolved settings instance."""
    if APP_SETTINGS_INSTANCE is None:
        raise RuntimeError(
            "Configuration not yet initialized! "
            "Call load_settings() from the application entry point first."
        )
    return APP_SETTINGS_INSTANCE


def _determine_env_files(app_env: str) -> list[pathlib.Path]:
    """Internal helper to conditionally determine the correct files."""
    env_files = []

    # 1. Add the environment-specific file
    if app_env == "dev":
        env_files.append(PROJECT_ROOT / "config" / ".env.dev")
    elif app_env == "staging":
        env_files.append(PROJECT_ROOT / "config" / ".env.staging")
    # Production uses OS ENV only, so no file added here.

    # 2. Add the local override file (highest file precedence)
    env_files.append(PROJECT_ROOT / ".env")

    return env_files


def load_settings(**kwargs: Any) -> AppSettings:
    """
    The ONLY place where settings are instantiated and resolved.
    This function must be called exactly once by run_server.py.
    """
    global APP_SETTINGS_INSTANCE

    if APP_SETTINGS_INSTANCE is not None:
        # Prevent re-instantiation
        return APP_SETTINGS_INSTANCE

    app_env = (
        kwargs.get("ENV")
        or os.environ.get("APP_ENV")
        or AppSettings.model_fields["ENV"].default
    )
    resolved_env_files = _determine_env_files(app_env)
    print(f"Loading configuration files for '{app_env}': {resolved_env_files}")

    # Copy the existing model_config (prefix, encoding, etc.)
    config_dict = AppSettings.model_config.copy()

    # Add the dynamically resolved env_file list to the configuration
    config_dict["env_file"] = resolved_env_files

    # Create a temporary BaseSettings class using the original fields
    # and the new configuration dictionary.
    class RuntimeSettings(AppSettings):
        model_config = config_dict

    # Instantiate the temporary class
    # We pass ONLY the CLI overrides to the constructor.
    # RuntimeSettings automatically loads the files defined in model_config.
    APP_SETTINGS_INSTANCE = RuntimeSettings(**kwargs)

    settings_json = APP_SETTINGS_INSTANCE.model_dump_json(indent=2)
    print("--- Final Configuration (JSON) ---")
    print(settings_json)

    return APP_SETTINGS_INSTANCE
