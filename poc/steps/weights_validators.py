import os

from poc.step import step


@step
def weights_validator(context: dict, weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights {weights_path} not found")
    if "cls" not in weights_path:
        raise ValueError(f"weights {weights_path} is not a classification weights")
    return weights_path
