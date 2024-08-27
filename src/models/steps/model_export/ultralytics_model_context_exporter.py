import os
import shutil

from picsellia import Experiment

from src.models.model.ultralytics.ultralytics_model_context import (
    UltralyticsModelContext,
)
from src.models.steps.model_export.model_context_exporter import ModelContextExporter


class UltralyticsModelContextExporter(ModelContextExporter):
    def __init__(self, model_context: UltralyticsModelContext, experiment: Experiment):
        super().__init__(model_context=model_context, experiment=experiment)

    def export_model_context(self):
        self.model_context.loaded_model.export(format="onnx")
        results_dir = os.listdir(self.model_context.results_path)
        if len(results_dir) == 0:
            raise ValueError("No results folder found")
        elif len(results_dir) == 1:
            model_folder = results_dir[0]
        else:
            model_folder = sorted(
                [
                    f
                    for f in results_dir
                    if f.startswith(self.model_context.model_name) and f[-1].isdigit()
                ]
            )[-1]
        exported_model_dir = os.path.join(
            self.model_context.results_path, model_folder, "weights"
        )
        onnx_file = [f for f in os.listdir(exported_model_dir) if f.endswith(".onnx")]
        if len(onnx_file) == 0:
            raise ValueError("No onnx file found")
        shutil.move(
            os.path.join(exported_model_dir, onnx_file[0]),
            self.model_context.inference_model_path,
        )
