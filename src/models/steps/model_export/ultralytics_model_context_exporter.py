import os
import shutil

from picsellia import Experiment

from src.models.model.ultralytics_model_context import UltralyticsModelContext
from src.models.steps.model_export.model_context_exporter import ModelContextExporter


class UltralyticsModelContextExporter(ModelContextExporter):
    def __init__(self, model_context: UltralyticsModelContext, experiment: Experiment):
        super().__init__(model_context=model_context, experiment=experiment)

    def export_model_context(self):
        self.model_context.loaded_model.export(format="onnx")
        # move onnx to inference path
        exported_model_dir = os.path.join(
            self.model_context.results_path, self.model_context.model_name, "weights"
        )
        # find the onnx file
        onnx_file = [f for f in os.listdir(exported_model_dir) if f.endswith(".onnx")]
        if len(onnx_file) == 0:
            raise ValueError("No onnx file found")
        shutil.move(
            os.path.join(exported_model_dir, onnx_file[0]),
            self.model_context.inference_model_path,
        )
