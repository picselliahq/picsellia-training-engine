from pathlib import Path
from typing import List
from ultralytics.models.yolo.classify import (
    ClassificationTrainer,
    ClassificationValidator,
)
from picsellia import Experiment

from src.models.steps.model_logging.ultralytics_classification_logger import (
    UltralyticsClassificationLogger,
)


class UltralyticsCallbacks:
    def __init__(self, experiment: Experiment):
        self.logger = UltralyticsClassificationLogger(experiment)

    def on_train_epoch_end(self, trainer: ClassificationTrainer):
        """Logs current training progress."""
        for metric_name, loss_value in trainer.label_loss_items(trainer.tloss).items():
            if metric_name.startswith("val"):
                self.logger.log_metric(
                    name=metric_name, value=float(loss_value), phase="val"
                )
            else:
                self.logger.log_metric(
                    name=metric_name, value=float(loss_value), phase="train"
                )

        for lr_name, lr_value in trainer.lr.items():
            self.logger.log_metric(name=lr_name, value=float(lr_value), phase="train")

    def on_fit_epoch_end(self, trainer: ClassificationTrainer):
        """Reports model information to logger at the end of an epoch."""
        self.logger.log_metric(
            name="epoch_time", value=float(trainer.epoch_time), phase="train"
        )

        for metric_name, metric_value in trainer.metrics.items():
            if metric_name.startswith("val"):
                self.logger.log_metric(
                    name=metric_name, value=float(metric_value), phase="val"
                )
            else:
                self.logger.log_metric(
                    name=metric_name, value=float(metric_value), phase="train"
                )

        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for info_key, info_value in model_info_for_loggers(trainer).items():
                self.logger.log_value(name=info_key, value=info_value)

    def on_val_end(self, validator: ClassificationValidator):
        """Logs validation results including labels and predictions."""
        validation_images_directory = Path(validator.save_dir)
        image_files = sorted(validation_images_directory.glob("val*.jpg"))
        for image_file in image_files:
            self.logger.log_image(
                name=image_file.stem.replace("val_", ""),
                image_path=str(image_file),
                phase="val",
            )

    def on_train_end(self, trainer: ClassificationTrainer):
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
            self.logger.log_image(
                name=file_path.stem, image_path=str(file_path), phase="val"
            )

        for metric_key, metric_value in trainer.validator.metrics.results_dict.items():
            self.logger.log_value(name=metric_key, value=metric_value, phase="val")

    def get_callbacks(self):
        return {
            "on_train_epoch_end": self.on_train_epoch_end,
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_val_end": self.on_val_end,
            "on_train_end": self.on_train_end,
        }
