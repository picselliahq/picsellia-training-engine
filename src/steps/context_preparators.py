from picsellia import Client

from src import step


@step
def context_preparator(
    api_token: str, host: str, organization_name: str, experiment_id: str
):
    client = Client(api_token=api_token, host=host, organization_name=organization_name)
    experiment = client.get_experiment_by_id(experiment_id)
    training_args = {
        "epochs": 1,
        "batch": 4,
        "imgsz": 224,
        "device": "mps",
        "cache": "ram",
        "deterministic": False,
        "seed": 0,
        "save_period": 1,
        "val": False,
    }
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
    exporter_args = {
        "format": "onnx",
        "device": "mps",
        "half": False,
        "dynamic": False,
        "imgsz": 224,
        "batch": 1,
        "optimize": False,
        "int8": False,
    }
    inference_args = {
        "batch": 1,
        "imgsz": 224,
        "device": "mps",
        "cache": "ram",
        "deterministic": False,
        "seed": 0,
    }
    context = {
        "client": client,
        "experiment": experiment,
        "training_args": training_args,
        "augmentation_args": augmentation_args,
        "exporter_args": exporter_args,
        "inference_args": inference_args,
    }
    return context
