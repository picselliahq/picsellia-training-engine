import logging
import os
import subprocess

import yaml
from picsellia import Experiment

from src.models.model.common.model_context import ModelContext
from src.models.steps.model_export.model_context_exporter import ModelContextExporter

logger = logging.getLogger(__name__)


class PaddleOCRModelContextExporter(ModelContextExporter):
    def __init__(self, model_context: ModelContext, experiment: Experiment):
        super().__init__(model_context=model_context, experiment=experiment)
        self.config = self.get_config()
        self.current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f".:{self.current_pythonpath}"

    def get_config(self) -> dict:
        if not self.model_context.config_path:
            raise ValueError("No configuration file path found in model context")
        with open(self.model_context.config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

    def write_config(self):
        if not self.model_context.config_path:
            raise ValueError("No configuration file path found in model context")
        with open(self.model_context.config_path, "w") as file:
            yaml.dump(self.config, file)

    def find_model_path(self, saved_model_path: str):
        model_files = [
            f
            for f in os.listdir(saved_model_path)
            if os.path.isfile(os.path.join(saved_model_path, f))
        ]
        for model_file in model_files:
            if isinstance(model_file, str):
                if model_file.startswith("best_accuracy"):
                    return os.path.join(saved_model_path, "best_accuracy")
                if model_file.startswith("latest"):
                    return os.path.join(saved_model_path, "latest")
        return None

    def export_model(self):
        if not self.model_context.config_path:
            raise ValueError("No configuration file path found in model context")
        command = [
            "python3.10",
            "src/pipelines/paddle_ocr/PaddleOCR/tools/export_model.py",
            "-c",
            self.model_context.config_path,
        ]

        os.setuid(os.geteuid())

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        process.wait()
        if process.returncode != 0:
            print("Export failed with errors")
            if process.stderr:
                errors = process.stderr.read()
                print(errors)

        os.environ["PYTHONPATH"] = self.current_pythonpath

    def export_model_context(
        self, exported_model_destination_path: str, export_format: str
    ):
        saved_model_dir = self.model_context.trained_weights_dir
        if not saved_model_dir:
            raise ValueError("No trained weights directory found in model context")

        found_model_path = self.find_model_path(saved_model_dir)

        if not found_model_path:
            logger.info(f"No model found in {saved_model_dir}, skipping export...")
        else:
            self.config["Global"]["pretrained_model"] = found_model_path
            self.config["Global"][
                "save_inference_dir"
            ] = exported_model_destination_path
            self.write_config()
            self.export_model()

        exported_model = os.listdir(exported_model_destination_path)
        if not exported_model:
            raise ValueError("No model files found in the exported model directory")
        else:
            logger.info(f"Model exported to {exported_model_destination_path}")
