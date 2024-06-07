from ultralytics import YOLO

from src import step


@step
def model_loader(
    weights_path: str,
):
    model = YOLO(model=weights_path, task="classify")
    return model
