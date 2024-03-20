from src import step


@step
def augmentation_preparator():
    augmentation_args = {
        "scale": 0.92,
        "fliplr": 0.5,
        "flipud": 0.0,
        "auto_augment": False,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.4,
        "erasing": 0.0,
    }
    return augmentation_args
