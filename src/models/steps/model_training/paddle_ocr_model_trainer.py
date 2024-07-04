from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection

import subprocess


def train_model(config_path):
    """
    Trains a Paddle OCR model given a configuration path and captures the output metrics line by line.

    Args:
        config_path (str): The path to the model's YAML configuration file.
    """
    # Set up the environment variable for PYTHONPATH
    import os

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f".:{current_pythonpath}"

    command = [
        "python3",
        "src/pipelines/paddle_ocr/PaddleOCR/tools/train.py",
        "-c",
        config_path,
    ]

    # Execute the command and capture output in real-time
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if "epoch:" in line:
                    extract_and_log_metrics(line)
    except Exception as e:
        print("Error during model training:", e)

    # Wait for the process to terminate and get the exit code
    process.wait()
    if process.returncode != 0:
        print("Training failed with errors")
        if process.stderr:
            errors = process.stderr.read()
            print(errors)

    # Reset PYTHONPATH if necessary
    os.environ["PYTHONPATH"] = current_pythonpath


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
            except ValueError:
                metrics[key] = value  # Keep as string if it can't be converted

    print("Metrics Parsed:", metrics)
    # Optionally, send these metrics to your monitoring or logging platform
    # log_metrics_to_platform(metrics)


class PaddleOCRModelTrainer:
    def __init__(self, model_collection: PaddleOCRModelCollection):
        self.model_collection = model_collection

    def train(self):
        """
        Trains both the bounding box detection and text recognition models in the model collection.
        """
        print("Starting training for bounding box model...")
        train_model(self.model_collection.bbox_model.config_file_path)

        print("Starting training for text recognition model...")
        train_model(self.model_collection.text_model.config_file_path)

        return self.model_collection
