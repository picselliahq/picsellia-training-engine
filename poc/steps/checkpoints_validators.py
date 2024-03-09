import os

from poc.step import step


@step
def checkpoints_validator(context: dict, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    if "cls" not in checkpoint_path:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a classification checkpoint"
        )
    return checkpoint_path
