from pathlib import Path
from typing import List

from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics.models.yolo.classify import (
    ClassificationTrainer,
    ClassificationValidator,
)

from src.models.contexts.picsellia_context import PicselliaTrainingContext
from src import step, Pipeline


def on_train_epoch_end(trainer: ClassificationTrainer, experiment: Experiment):
    """Logs current training progress."""
    for metric_name, loss_value in trainer.label_loss_items(
        trainer.tloss, prefix="train"
    ).items():
        experiment.log(metric_name, float(loss_value), LogType.LINE)
    for lr_name, lr_value in trainer.lr.items():
        experiment.log(lr_name, float(lr_value), LogType.LINE)


def on_fit_epoch_end(trainer: ClassificationTrainer, experiment: Experiment):
    """Reports model information to logger at the end of an epoch."""
    experiment.log("epoch_time(s)", float(trainer.epoch_time), LogType.LINE)
    for metric_name, metric_value in trainer.metrics.items():
        experiment.log(metric_name, float(metric_value), LogType.LINE)
    if trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        for info_key, info_value in model_info_for_loggers(trainer).items():
            experiment.log(info_key, info_value, LogType.VALUE)


def on_val_end(validator: ClassificationValidator, experiment: Experiment):
    """Logs validation results including labels and predictions."""
    validation_images_directory = Path(validator.save_dir)
    image_files = sorted(validation_images_directory.glob("val*.jpg"))
    for image_file in image_files:
        experiment.log(image_file.stem, str(image_file), LogType.IMAGE)


def on_train_end(trainer: ClassificationTrainer, experiment: Experiment):
    """Logs final model and its name on training completion."""
    model_output_directory = Path(trainer.save_dir)
    visualization_files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        *(f"{metric}_curve.png" for metric in ("F1", "PR", "P", "R")),
    ]
    existing_files: List[Path] = [
        model_output_directory / file_name
        for file_name in visualization_files
        if (model_output_directory / file_name).exists()
    ]
    for file_path in existing_files:
        experiment.log(file_path.stem, str(file_path), LogType.IMAGE)
    # Report final metrics
    for metric_key, metric_value in trainer.validator.metrics.results_dict.items():
        experiment.log(f"final_val/{metric_key}", metric_value, LogType.VALUE)


@step
def callback_preparator():
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    return {
        "on_train_epoch_end": lambda trainer: on_train_epoch_end(
            trainer, context.experiment
        ),
        "on_fit_epoch_end": lambda trainer: on_fit_epoch_end(
            trainer, context.experiment
        ),
        "on_val_end": lambda validator: on_val_end(validator, context.experiment),
        "on_train_end": lambda trainer: on_train_end(trainer, context.experiment),
    }
