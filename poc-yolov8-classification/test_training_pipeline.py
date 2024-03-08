import argparse
import os

from picsellia import Client, DatasetVersion, Artifact
from ultralytics.models.yolo.classify import ClassificationTrainer

from utils import DatasetOrganizer


def context_preparator(
    api_token: str, host: str, organization_name: str, experiment_id: str
):
    client = Client(api_token=api_token, host=host, organization_name=organization_name)
    experiment = client.get_experiment_by_id(experiment_id)
    parameters = {
        "epochs": 3,
        "batch": 4,
        "imgsz": 640,
        "device": "cuda",
        "cache": "ram",
    }
    context = {"client": client, "experiment": experiment, "parameters": parameters}
    return context


def dataset_extractor(dataset: DatasetVersion) -> dict:
    dataset_extraction_path = dataset.name
    dataset.download(
        target_path=os.path.join(dataset_extraction_path, "images"), use_id=True
    )
    coco_file = dataset.build_coco_file_locally(use_id=True)
    labelmap = {str(i): label.name for i, label in enumerate(dataset.list_labels())}
    train_dataset_context = {
        "dataset_extraction_path": dataset_extraction_path,
        "images_dir": os.path.join(dataset_extraction_path, "images"),
        "coco_file": coco_file,
        "labelmap": labelmap,
    }
    return train_dataset_context


def data_extractor(context: dict):
    train_dataset: DatasetVersion = context["experiment"].get_dataset("train")
    test_dataset: DatasetVersion = context["experiment"].get_dataset("test")
    train_dataset_context = dataset_extractor(train_dataset)
    test_dataset_context = dataset_extractor(test_dataset)
    return train_dataset_context, test_dataset_context


def augmentation_preparator(context: dict):
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


def data_preparator(
    context: dict, train_dataset_context: dict, test_dataset_context: dict
):
    train_converter = DatasetOrganizer(
        train_dataset_context["coco_file"],
        train_dataset_context["images_dir"],
        os.path.join(train_dataset_context["dataset_extraction_path"], "train"),
    )
    train_converter.organize()

    test_converter = DatasetOrganizer(
        test_dataset_context["coco_file"],
        test_dataset_context["images_dir"],
        os.path.join(test_dataset_context["dataset_extraction_path"], "test"),
    )
    test_converter.organize()

    if (
        train_dataset_context["dataset_extraction_path"]
        == test_dataset_context["dataset_extraction_path"]
    ):
        return os.path.abspath(train_dataset_context["dataset_extraction_path"])
    else:
        raise ValueError("Train and test global dataset paths are different")


def checkpoints_extractor(context: dict):
    model_file: Artifact = context["experiment"].get_artifact("weights")
    model_file.download(
        target_path=os.path.join(context["experiment"].name, "checkpoints")
    )
    return os.path.abspath(
        os.path.join(context["experiment"].name, "checkpoints", model_file.filename)
    )


def checkpoints_preparator(context: dict, checkpoints_path: str):
    return checkpoints_path


def checkpoints_validator(context: dict, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    if "cls" not in checkpoint_path:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a classification checkpoint"
        )
    return checkpoint_path


def model_loader(
    context: dict, checkpoints_path: str, dataset_path: str, augmentation_args: dict
):
    model_args = dict(
        model=checkpoints_path,
        project=context["experiment"].name,
        data=dataset_path,
        **context["parameters"],
        **augmentation_args,
    )
    print(f"model_args: {model_args}")
    trainer = ClassificationTrainer(overrides=model_args)
    return trainer


def model_trainer(trainer: ClassificationTrainer):
    trainer.train()
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--api_token", type=str, help="API token")
    parser.add_argument("--organization_name", type=str, help="Organization name")
    parser.add_argument(
        "--host",
        type=str,
        default="https://app.picsellia.com",
        required=False,
        help="Host",
    )
    parser.add_argument("--experiment_id", type=str, help="Experiment id")
    args = parser.parse_args()

    context = context_preparator(
        args.api_token, args.host, args.organization_name, args.experiment_id
    )
    train_dataset_context, test_dataset_context = data_extractor(context)
    augmentation_args = augmentation_preparator(context)
    data_path = data_preparator(context, train_dataset_context, test_dataset_context)
    checkpoints_path = checkpoints_extractor(context)
    checkpoints_path = checkpoints_preparator(context, checkpoints_path)
    checkpoints_path = checkpoints_validator(context, checkpoints_path)
    trainer = model_loader(context, checkpoints_path, data_path, augmentation_args)
    model_trainer(trainer)
