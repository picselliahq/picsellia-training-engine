import logging
import os

from trainer import YoloxDetectionTrainer

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
logging.getLogger("picsellia").setLevel(logging.INFO)

training_pipeline = YoloxDetectionTrainer()
training_pipeline.prepare_data_for_training()
training_pipeline.train()
training_pipeline.eval()
