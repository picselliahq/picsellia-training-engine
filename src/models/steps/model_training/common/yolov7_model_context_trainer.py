import os
import subprocess
import sys
import time

from picsellia import Experiment

from src.models.dataset.common.yolov7_dataset_collection import Yolov7DatasetCollection
from src.models.model.common.model_context import ModelContext
from src.models.model.yolov7_model_context import Yolov7ModelContext
from src.models.parameters.training.yolov7.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)


def handle_training_failure(process: subprocess.Popen):
    """
    Handles training failure by printing error messages from the training process.

    Args:
        process (subprocess.Popen): The process object running the training command.
    """
    print("Training failed with errors")
    if process.stderr:
        errors = process.stderr.read()
        print(errors)


class Yolov7ModelContextTrainer:
    def __init__(self, model_context: Yolov7ModelContext, experiment: Experiment):
        """
        Initializes the trainer with a model context and experiment.

        Args:
            model_context (ModelContext): The context for the PaddleOCR model being trained.
            experiment (Experiment): The Picsellia experiment to log training metrics.
        """
        self.model_context = model_context
        self.experiment = experiment

    def train_model_context(
        self,
        dataset_collection: Yolov7DatasetCollection,
        hyperparameters: Yolov7HyperParameters,
        api_token: str,
        organization_id: str,
        host: str,
        experiment_id: str,
    ):
        # command = [
        #     "python3.10",
        #
        #     "src/pipelines/yolov7_segmentation/yolov7/train.py",
        #
        #     "--weights",
        #     self.model_context.pretrained_weights_path,
        #
        #     "--cfg",
        #     self.model_context.config_path,
        #
        #     "--data",
        #     dataset_collection.config_path,
        #
        #     "--hyp",
        #     self.model_context.hyperparameters_path,
        #
        #     "--epochs",
        #     str(hyperparameters.epochs),
        #
        #     "--batch-size",
        #     str(hyperparameters.batch_size),
        #
        #     "--img-size",
        #     str(hyperparameters.image_size),
        #
        #     "--device",
        #     str(hyperparameters.device),
        #
        #     "--project",
        #     self.model_context.results_dir,
        #
        #     "--name",
        #     self.model_context.model_name,
        # ]

        command = [
            "python3.10",
            "src/pipelines/yolov7_segmentation/yolov7/train.py",
            "--img-size",
            "640",
            "--cfg",
            "test_2/model/weights/config/yolov7.yaml",
            "--hyp",
            "test_2/model/weights/hyp.scratch.custom.yaml",
            "--batch-size",
            "4",
            "--epochs",
            "1",
            "--data",
            "test_2/yolov7_dataset/dataset_config.yaml",
            "--weights",
            "test_2/model/weights/pretrained_weights/yolov7_training.pt",
            "--name",
            "Yolov7-0",
            "--api_token",
            f"{api_token}",
            "--organization_id",
            f"{organization_id}",
            "--host",
            f"{host}",
            "--experiment_id",
            f"{experiment_id}",
        ]

        print(f"Command: {command}")

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

        # process = subprocess.run(
        #     command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        # )
        #
        # print("Standard Output:\n", process.stdout)
        # print("Error Output:\n", process.stderr)
        # print("Exit Code:", process.returncode)
        #
        # if process.returncode != 0:
        #     print("An error occurred during training")

    def _process_training_output(
        self, process: subprocess.Popen, model_context: ModelContext
    ):
        """
        Processes the output from the training subprocess and extracts metrics.

        This method reads the training logs line by line, extracts relevant metrics, and logs
        them to the Picsellia experiment. Only new metrics from previously unlogged epochs are recorded.

        Args:
            process (subprocess.Popen): The subprocess object running the training command.
            model_context (ModelContext): The model context containing information about the model being trained.
        """
        if process.stdout:
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                # Flush output to ensure continuous printing in real-time
                sys.stdout.flush()
                time.sleep(0.1)  # Small delay to avoid excessive CPU usage
