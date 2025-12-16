import os
import argparse
import uvicorn
from .setup_logging import load_logging_config, setup_logging
from .settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the term subscription prediction API server."
    )
    parser.add_argument("--host", help="Host IP to bind to.")
    parser.add_argument("--port", type=int, help="Port to listen on.")
    parser.add_argument(
        "--workers", type=int, help="Number of worker processes."
    )
    parser.add_argument(
        "--env", choices=['dev', 'staging', 'prod'], help="Deployment environment."
    )

    args = parser.parse_args()

    override_kwargs = {k.upper(): v for k, v in vars(args).items() if v is not None}
    settings = load_settings(**override_kwargs)

    logging_config_dict = load_logging_config(env=settings.ENV)
    setup_logging(logging_config_dict)

    # if settings.ENV:
    #     uvicorn.run("api_server.main:app", host=settings.HOST, port=settings.PORT, reload=True, log_config=logging_config_dict)
    # else:
    #     uvicorn.run(
    #         "api_server.main:app", host=settings.HOST, port=settings.PORT, workers=settings.WORKERS, log_config=logging_config_dict
    #     )


if __name__ == "__main__":
    main()
