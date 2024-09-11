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
        """
        Initializes the UltralyticsModelContextExporter.

        Args:
            model_context (ModelContext): The model context containing information about the model and its paths.
            experiment (Experiment): The experiment associated with the model context.
        """
        super().__init__(model_context=model_context, experiment=experiment)

    def export_model_context(
        self, exported_weights_destination_path: str, export_format: str
    ) -> None:
        """
        Exports the Ultralytics model context by converting it to the specified format (typically ONNX)
        and moves the resulting file to the specified destination path.

        Args:
            exported_weights_destination_path (str): The path where the exported model weights should be saved.
            export_format (str): The format to export the model (e.g., ONNX).

        Raises:
            ValueError: If no results folder or ONNX file is found.
        """
        self._export_model(export_format=export_format)

        onnx_file_path = self._find_exported_onnx_file()

        self._move_onnx_to_destination_path(
            onnx_file_path=onnx_file_path,
            exported_weights_destination_path=exported_weights_destination_path,
        )

    def _export_model(self, export_format: str) -> None:
        """
        Exports the loaded model in the specified format (e.g., ONNX) to the inference model path.

        Args:
            export_format (str): The format to export the model in.
        """
        self.model_context.loaded_model.export(format=export_format)

    def _find_ultralytics_results_dir(self) -> str:
        """
        Locates the appropriate results folder within the model's results directory.

        Returns:
            str: The full path to the results folder.

        Raises:
            ValueError: If no results folder is found in the results directory.
        """
        results_dirs = os.listdir(self.model_context.results_dir)
        if not results_dirs:
            raise ValueError("No results folder found")
        elif len(results_dirs) == 1:
            return os.path.join(self.model_context.results_dir, results_dirs[0])

        return os.path.join(
            self.model_context.results_dir,
            sorted(
                [
                    f
                    for f in results_dirs
                    if f.startswith(self.model_context.model_name) and f[-1].isdigit()
                ]
            )[-1],
        )

    def _find_exported_onnx_file(self) -> str:
        """
        Searches for the ONNX file in the weights directory of the Ultralytics results folder.

        Returns:
            str: The full path to the ONNX file.

        Raises:
            ValueError: If no ONNX file is found in the weights directory.
        """
        ultralytics_results_dir = self._find_ultralytics_results_dir()
        ultralytics_weights_dir = os.path.join(ultralytics_results_dir, "weights")
        onnx_files = [
            f for f in os.listdir(ultralytics_weights_dir) if f.endswith(".onnx")
        ]
        if not onnx_files:
            raise ValueError("No ONNX file found")
        return os.path.join(ultralytics_weights_dir, onnx_files[0])

    def _move_onnx_to_destination_path(
        self, onnx_file_path: str, exported_weights_destination_path: str
    ) -> None:
        """
        Moves the ONNX file from its current location to the specified destination path.

        Args:
            onnx_file_path (str): The full path to the ONNX file.
            exported_weights_destination_path (str): The destination path where the ONNX file should be moved.

        """
        print(f"Moving ONNX file to {exported_weights_destination_path}...")
        shutil.move(onnx_file_path, exported_weights_destination_path)
