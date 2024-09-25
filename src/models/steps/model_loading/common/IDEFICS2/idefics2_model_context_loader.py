import logging

import torch
from transformers import AutoModelForVision2Seq

logger = logging.getLogger(__name__)


def idefics2_load_model(model_name: str, device: str):
    model = AutoModelForVision2Seq.from_pretrained(model_name)

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
