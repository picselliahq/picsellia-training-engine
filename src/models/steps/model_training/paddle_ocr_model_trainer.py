import os

from src.models.model.model_context import ModelContext
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection

import subprocess

from picsellia import Experiment
from picsellia.sdk.log import LogType


def extract_and_log_metrics(log_line):
    """
    Extract metrics from a log line by stripping unnecessary prefixes and parsing key-value pairs.

    Args:
        log_line (str): A single line of log output from the training process.
    """
    # Remove the timestamp and logger info
    log_line = log_line.split("ppocr INFO:")[-1].strip()

    # Now, split the log line on commas to get each metric as a key-value pair
    metrics = {}
    key_value_pairs = log_line.split(",")
    for pair in key_value_pairs:
        if ":" in pair:
            key, value = pair.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                # Convert numeric values
                if "." in value:
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)
                if key == "epoch":
                    metrics[key] = int(value.replace("[", "").split("/")[0])
            except ValueError:
                metrics[key] = value  # Keep as string if it can't be converted

    return metrics


class PaddleOCRModelTrainer:
    def __init__(
        self, model_collection: PaddleOCRModelCollection, experiment: Experiment
    ):
        self.model_collection = model_collection
        self.experiment = experiment

    def train(self):
        """
        Trains both the bounding box detection and text recognition models in the model collection.
        """
        print("Starting training for bounding box model...")
        self.train_model(model_context=self.model_collection.bbox_model)

        print("Starting training for text recognition model...")
        self.train_model(model_context=self.model_collection.text_model)

        return self.model_collection

    def train_model(self, model_context: ModelContext):
        """
        Trains a Paddle OCR model given a configuration path and captures the output metrics line by line.

        Args:
            prefix_model_name: The prefix of the model name to train (either "bbox" or "text").
        """
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f".:{current_pythonpath}"

        config_path = model_context.config_file_path
        if not config_path:
            raise ValueError("No configuration file path found in model context")

        command = [
            "python3",
            "src/pipelines/paddle_ocr/PaddleOCR/tools/train.py",
            "-c",
            config_path,
        ]

        joined_command = " ".join(command)

        process = subprocess.Popen(
            joined_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        try:
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if "epoch:" in line:
                        metrics = extract_and_log_metrics(line)
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                self.experiment.log(
                                    name=f"{model_context.prefix_model_name}_{key}",
                                    data=value,
                                    type=LogType.LINE,
                                )
        except Exception as e:
            print("Error during model training:", e)

        process.wait()
        if process.returncode != 0:
            print("Training failed with errors")
            if process.stderr:
                errors = process.stderr.read()
                print(errors)

        os.environ["PYTHONPATH"] = current_pythonpath
