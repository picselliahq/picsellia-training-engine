import os
import subprocess

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

        command = [
            "python3",
            "src/pipelines/paddle_ocr/PaddleOCR/tools/export_model.py",
            "-c",
            config_path,
        ]

        joined_command = " ".join(command)

        process = subprocess.Popen(
            joined_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        process.wait()
        if process.returncode != 0:
            print("Export failed with errors")
            if process.stderr:
                errors = process.stderr.read()
                print(errors)

        os.environ["PYTHONPATH"] = current_pythonpath
