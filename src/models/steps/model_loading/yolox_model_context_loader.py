from typing import Dict

import torch
from picsellia import Label

from src.models.steps.model_loading.architecture.yolox.yolox_base import YoloX


def load_yolox_weights(
    model_path: str, model_architecture: str, labelmap: Dict[str, Label], device: str
) -> torch.nn.Module:
    """
    Load a PyTorch model from a file.

    Args:
        model_path (str): Path to the saved model file.
        model_architecture (str): The architecture of the model to load.
        labelmap (Dict[str, Label]): Labelmap for the model.
        device (str): Device to load the model on ('cpu', 'cuda', or 'mps').

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    model = YoloX(architecture=model_architecture, labelmap=labelmap).get_model()
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()

    return model
