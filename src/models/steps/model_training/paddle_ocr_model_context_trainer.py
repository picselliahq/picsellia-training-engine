import os
import subprocess
from typing import Dict, Union

from picsellia import Experiment
from picsellia.sdk.log import LogType

from src.models.model.common.model_context import ModelContext


def extract_and_log_metrics(log_line: str) -> Dict[str, Union[str, int, float]]:
    """
    Extract metrics from a log line by stripping unnecessary prefixes and parsing key-value pairs.

    Args:
        log_line (str): A single line of log output from the training process.

    Returns:
        Dict[str, Union[str, int, float]]: Extracted metrics as a dictionary.
    """
    log_line = log_line.split("ppocr INFO:")[-1].strip()
    metrics: Dict[str, Union[str, int, float]] = {}
    key_value_pairs = log_line.split(",")

    for pair in key_value_pairs:
        if ":" in pair:
            key, value = pair.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                if key == "epoch":
                    metrics[key] = int(value.replace("[", "").split("/")[0])
                elif "." in value:
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)
            except ValueError:
                metrics[key] = value

    return metrics


def handle_training_failure(process: subprocess.Popen):
    print("Training failed with errors")
    if process.stderr:
        errors = process.stderr.read()
        print(errors)


class PaddleOCRModelContextTrainer:
    def __init__(self, model_context: ModelContext, experiment: Experiment):
        self.model_context = model_context
        self.experiment = experiment
        self.last_logged_epoch: Union[int, None] = None  # Last epoch that was logged

    def train_model_context(self):
        """
        Trains a Paddle OCR model given a configuration path and captures the output metrics line by line.

        Args:
            model_context (ModelContext): The model context containing the configuration path.
        """
        print(f"Starting training for {self.model_context.prefix_model_name} model...")

        config_path = self.model_context.config_file_path
        if not config_path:
            raise ValueError(
                f"No configuration file path found in {self.model_context.prefix_model_name} model context"
            )

        command = [
            "python3.10",
            "src/pipelines/paddle_ocr/PaddleOCR/tools/train.py",
            "-c",
            config_path,
        ]

        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f".:{current_pythonpath}"
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        try:
            self._process_training_output(process, self.model_context)
        except Exception as e:
            print("Error during model training:", e)
        finally:
            process.wait()
            if process.returncode != 0:
                handle_training_failure(process)

            os.environ["PYTHONPATH"] = current_pythonpath

    def _process_training_output(
        self, process: subprocess.Popen, model_context: ModelContext
    ):
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line.strip())
                if "epoch:" in line:
                    metrics = extract_and_log_metrics(line)
                    current_epoch = metrics.get("epoch")
                    if (
                        current_epoch is not None
                        and isinstance(current_epoch, int)
                        and current_epoch != self.last_logged_epoch
                    ):
                        self.last_logged_epoch = current_epoch
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if k not in ["epoch", "global_step"]
                        }
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                self.experiment.log(
                                    name=f"{model_context.prefix_model_name}/{key}",
                                    data=value,
                                    type=LogType.LINE,
                                )
