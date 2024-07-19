import os
import subprocess

import yaml
from picsellia import Experiment

from src.models.model.model_context import ModelContext
from src.models.steps.model_export.model_context_exporter import ModelContextExporter


class PaddleOCRModelContextExporter(ModelContextExporter):
    def __init__(self, model_context: ModelContext, experiment: Experiment):
        super().__init__(model_context=model_context, experiment=experiment)

    def export_model(self):
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f".:{current_pythonpath}"

        config_path = self.model_context.config_file_path
        if not config_path:
            raise ValueError("No configuration file path found in model context")

        with open(config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        save_model_dir = config["Global"]["save_model_dir"]

        found_best_accuracy = False
        found_latest = False
        if config["Global"]["save_model_dir"]:
            for file in os.listdir(save_model_dir):
                if file.startswith("best_accuracy"):
                    found_best_accuracy = True
                if file.startswith("latest"):
                    found_latest = True

        if not found_best_accuracy and not found_latest:
            print(f"No model found in {save_model_dir}, skipping export...")
        elif not found_best_accuracy:
            config["Global"]["pretrained_model"] = f"{save_model_dir}/latest"
        else:
            config["Global"]["pretrained_model"] = f"{save_model_dir}/best_accuracy"

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
