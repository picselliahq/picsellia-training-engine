import logging

import torch
from transformers import CLIPModel

logger = logging.getLogger(__name__)


def clip_load_model(device: str):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    model.eval()

    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("Using CPU")
        device = torch.device("cpu")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    model.to(device)
    return model
