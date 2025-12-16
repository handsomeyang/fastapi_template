import argparse
import json
import requests
import pandas as pd
from utils import get_data_dir
from colorama import init, Fore, Style
from .settings import get_settings


init()


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Query the term subscription prediction endpoint."
    )
    parser.add_argument("--host", default=settings.HOST, help="API server address.")
    parser.add_argument(
        "--port", type=int, default=settings.PORT, help="API server port."
    )
    parser.add_argument("--data", help="Customer data in JSON string format.")

    args = parser.parse_args()

    headers = {"Content-Type": "application/json"}
    api_url = f"http://{args.host}:{args.port}/predict"

    ground_truth = None
    if args.data is None:
        print(Fore.YELLOW + "No input data. Sampling the dataset.\n" + Style.RESET_ALL)

        dataset_path = get_data_dir() / "dataset.csv"
        try:
            df = pd.read_csv(dataset_path, sep=";")
            ground_truth = df.sample(n=1)
            user_data = json.loads(ground_truth.drop(columns=["y"]).iloc[0].to_json())

        except FileNotFoundError:
            print(
                Fore.RED
                + f"ERROR: Dataset file not found at {dataset_path}."
                + Style.RESET_ALL
            )
            return
    else:
        user_data = json.loads(args.data)

    print(Fore.GREEN + f"Sending data to {api_url}:" + Style.RESET_ALL)
    print(json.dumps(user_data, indent=2))
    print("\n")

    try:
        response = requests.post(api_url, json=user_data, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        print(Fore.CYAN + "Query result" + Style.RESET_ALL)
        print(json.dumps(response.json(), indent=2))

        if ground_truth is not None:
            print(Fore.CYAN + "\nGround truth" + Style.RESET_ALL)
            print(ground_truth["y"].iloc[0])

    except requests.exceptions.HTTPError as errh:
        print(Fore.RED + f"HTTP Error occurred: {errh}" + Style.RESET_ALL)
        print(f"Response body: {response.text}")
    except requests.exceptions.ConnectionError as errc:
        print(Fore.RED + f"Error Connecting: {errc}" + Style.RESET_ALL)
    except requests.exceptions.Timeout as errt:
        print(Fore.RED + f"Timeout Error: {errt}" + Style.RESET_ALL)
    except requests.exceptions.RequestException as err:
        print(Fore.RED + f"An unexpected error occurred: {err}" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
