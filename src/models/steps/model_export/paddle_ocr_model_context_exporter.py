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

    def export_model_context(
        self, exported_model_destination_path: str, export_format: str
    ):
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f".:{current_pythonpath}"

        config_path = self.model_context.config_path
        if not config_path:
            raise ValueError("No configuration file path found in model context")

        with open(config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        found_best_accuracy = False
        found_latest = False
        if exported_model_destination_path:
            model_files = [
                f
                for f in os.listdir(exported_model_destination_path)
                if os.path.isfile(os.path.join(exported_model_destination_path, f))
            ]
            for model_file in model_files:
                if isinstance(model_file, str):
                    if model_file.startswith("best_accuracy"):
                        found_best_accuracy = True
                    if model_file.startswith("latest"):
                        found_latest = True

        if not found_best_accuracy and not found_latest:
            logger.info(
                f"No model found in {exported_model_destination_path}, skipping export..."
            )
        elif not found_best_accuracy:
            config["Global"][
                "pretrained_model"
            ] = f"{exported_model_destination_path}/latest"
        else:
            config["Global"][
                "pretrained_model"
            ] = f"{exported_model_destination_path}/best_accuracy"

        with open(config_path, "w") as file:
            yaml.dump(config, file)

        if found_latest or found_best_accuracy:
            command = [
                "python3.10",
                "src/pipelines/paddle_ocr/PaddleOCR/tools/export_model.py",
                "-c",
                config_path,
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

            os.environ["PYTHONPATH"] = current_pythonpath
