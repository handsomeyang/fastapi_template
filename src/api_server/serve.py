import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the term subscription prediction API server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes."
    )
    parser.add_argument(
        "--dev", action="store_true", help="Enable Uvicorn hot-reloading."
    )

    args = parser.parse_args()

    if args.dev:
        uvicorn.run("api_server.main:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(
            "api_server.main:app", host=args.host, port=args.port, workers=args.workers
        )


if __name__ == "__main__":
    main()
