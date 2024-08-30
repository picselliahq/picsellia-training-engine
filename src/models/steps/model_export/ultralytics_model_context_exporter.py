import os
import shutil
from picsellia import Experiment

from src.models.model.common.model_context import ModelContext
from src.models.steps.model_export.model_context_exporter import ModelContextExporter


class UltralyticsModelContextExporter(ModelContextExporter):
    """
    Exporter class for Ultralytics model contexts.

    Attributes:
        model_context (ModelContext): The Ultralytics model context to be exported.
        experiment (Experiment): The experiment to which the model is related.
    """

    def __init__(self, model_context: ModelContext, experiment: Experiment):
        super().__init__(model_context=model_context, experiment=experiment)

    def export_model_context(
        self, exported_model_destination_path: str, export_format: str
    ) -> None:
        """
        Exports the Ultralytics model context by converting it to the ONNX format and ensuring
        that the resulting file is placed in the inference model path for subsequent saving.

        Raises:
            ValueError: If no results folder or ONNX file is found.
        """
        self._export_model(export_format=export_format)

        model_folder = self._find_model_folder()
        exported_model_dir = os.path.join(
            self.model_context.results_dir, model_folder, "weights"
        )
        onnx_file = self._find_onnx_file(exported_model_dir)

        self._move_onnx_to_destination_path(
            onnx_file_path=os.path.join(exported_model_dir, onnx_file),
            exported_model_destination_path=exported_model_destination_path,
        )

    def _export_model(self, export_format: str) -> None:
        """
        Exports the model to ONNX format directly into the inference model path.
        """
        self.model_context.loaded_model.export(format=export_format)

    def _find_model_folder(self) -> str:
        """
        Finds the correct model folder within the results directory.

        Returns:
            str: The name of the model folder.

        Raises:
            ValueError: If no results folder is found.
        """
        results_dir = os.listdir(self.model_context.results_dir)
        if not results_dir:
            raise ValueError("No results folder found")
        elif len(results_dir) == 1:
            return results_dir[0]

        return sorted(
            [
                f
                for f in results_dir
                if f.startswith(self.model_context.model_name) and f[-1].isdigit()
            ]
        )[-1]

    def _find_onnx_file(self, exported_model_dir: str) -> str:
        """
        Finds the ONNX file in the specified directory.

        Args:
            exported_model_dir (str): The directory to search for the ONNX file.

        Returns:
            str: The name of the ONNX file.

        Raises:
            ValueError: If no ONNX file is found.
        """
        onnx_files = [f for f in os.listdir(exported_model_dir) if f.endswith(".onnx")]
        if not onnx_files:
            raise ValueError("No ONNX file found")
        return onnx_files[0]

    def _move_onnx_to_destination_path(
        self, onnx_file_path: str, exported_model_destination_path: str
    ) -> None:
        """
        Moves the ONNX file to the specified destination path.

        Args:
            onnx_file_path:
            exported_model_destination_path:

        Returns:

        """
        print(
            f"Moving ONNX file to inference model path: {exported_model_destination_path}"
        )
        shutil.move(onnx_file_path, exported_model_destination_path)
