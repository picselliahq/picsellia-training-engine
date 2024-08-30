from ultralytics import YOLO


def ultralytics_load_model(weights_path_to_load: str, device: str) -> YOLO:
    """
    Loads the Ultralytics model using the pretrained model path.
    Raises an error if the pretrained model path is not set.
    """
    loaded_model = YOLO(weights_path_to_load)
    loaded_model.to(device)
    return loaded_model
