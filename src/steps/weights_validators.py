import os

from src import step


@step
def weights_validator(weights_path: str) -> str:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights {weights_path} not found.")

    if "cls" not in weights_path:
        raise ValueError(f"Weights {weights_path} are not classification weights.")

    return weights_path
