import os
import logging

from trainer import Yolov8ClassificationTrainer

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger("picsellia").setLevel(logging.INFO)

training_pipeline = Yolov8ClassificationTrainer()
training_pipeline.prepare_data_for_training()
experiment = training_pipeline.experiment


def on_train_epoch_end(trainer):
    metrics = trainer.metrics
    experiment.log("accuracy", float(metrics["metrics/accuracy_top1"]), "line")
    experiment.log("test/loss", float(metrics["test/loss"]), "line")


training_pipeline.model.add_callback("on_train_epoch_end", on_train_epoch_end)
training_pipeline.train()
training_pipeline.eval()
