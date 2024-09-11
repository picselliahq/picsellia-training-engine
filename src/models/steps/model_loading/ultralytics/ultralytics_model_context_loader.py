from ultralytics import YOLO
import torch


def ultralytics_load_model(weights_path_to_load: str, device: str) -> YOLO:
    """
    Loads the Ultralytics model using the pretrained model path.
    Raises an error if the pretrained model path is not set.
    """
    loaded_model = YOLO(weights_path_to_load)
    torch_device = torch.device(device)
    print(f"Loading model on device: {torch_device}")
    loaded_model.to(device=device)
    return loaded_model
