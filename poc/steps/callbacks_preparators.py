from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics.models.yolo.classify import (
    ClassificationTrainer,
    ClassificationValidator,
)

from poc.step import step


def on_train_epoch_end(trainer: ClassificationTrainer, experiment: Experiment):
    """Logs current training progress."""
    for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
        experiment.log(k, float(v), LogType.LINE)
    for k, v in trainer.lr.items():
        experiment.log(k, float(v), LogType.LINE)


def on_fit_epoch_end(trainer: ClassificationTrainer, experiment: Experiment):
    """Reports model information to logger at the end of an epoch."""
    experiment.log("epoch_time(s)", float(trainer.epoch_time), LogType.LINE)
    for k, v in trainer.metrics.items():
        experiment.log(k, float(v), LogType.LINE)
    if trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        for k, v in model_info_for_loggers(trainer).items():
            experiment.log(k, v, LogType.VALUE)


def on_val_end(validator: ClassificationValidator, experiment: Experiment):
    """Logs validation results including labels and predictions."""
    files = sorted(validator.save_dir.glob("val*.jpg"))
    for f in files:
        experiment.log(str(f.stem), str(f), LogType.IMAGE)


def on_train_end(trainer: ClassificationTrainer, experiment: Experiment):
    """Logs final model and its name on training completion."""
    # Log final results, CM matrix + PR plots
    files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
    ]
    files = [
        (trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()
    ]  # filter
    for f in files:
        experiment.log(str(f.stem), str(f), LogType.IMAGE)
    # Report final metrics
    for k, v in trainer.validator.metrics.results_dict.items():
        experiment.log(f"final_val/{k}", v, LogType.VALUE)
    # Log the final model


@step
def callback_preparator(context: dict):
    return {
        "on_train_epoch_end": lambda trainer: on_train_epoch_end(
            trainer, context["experiment"]
        ),
        "on_fit_epoch_end": lambda trainer: on_fit_epoch_end(
            trainer, context["experiment"]
        ),
        "on_val_end": lambda validator: on_val_end(validator, context["experiment"]),
        "on_train_end": lambda trainer: on_train_end(trainer, context["experiment"]),
    }
