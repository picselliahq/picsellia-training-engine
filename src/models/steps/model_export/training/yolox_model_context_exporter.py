import logging
import os
from enum import Enum

import coremltools as ct
import torch

from src.models.model.common.model_context import ModelContext
from src.models.steps.model_export.common.model_context_exporter import (
    ModelContextExporter,
)

logger = logging.getLogger(__name__)


class YoloXModelVersionConversionTargetFramework(Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    COREML = "coreml"


class YoloXModelContextExporter(ModelContextExporter):
    """
    Exporter class for YoloX model contexts.

    This class handles the exportation of models trained using the YoloX architecture. It exports the model to a specified format,
    typically ONNX, and moves the resulting file to a destination directory for further use or deployment.

    Attributes:
        model_context (ModelContext): The YoloX model context to be exported.
    """

    def __init__(self, model_context: ModelContext):
        """
        Initializes the UltralyticsModelContextExporter.

        Args:
            model_context (ModelContext): The model context containing information about the model and its paths.
        """
        super().__init__(model_context=model_context)
        model_version_parameters = model_context.model_version.sync()

        self.model_version_docker_env_variables = model_version_parameters[
            "docker_env_variables"
        ]
        self.model_version_base_parameters = model_version_parameters["base_parameters"]

    def export_model_context(
        self, exported_weights_destination_path: str, export_format: str
    ) -> str:
        """
        Exports the YoloX model context by converting it to the specified format (typically ONNX)
        and moves the resulting file to the specified destination path.

        Args:
            exported_weights_destination_path (str): The path where the exported model weights should be saved.
            export_format (str): The format to export the model (e.g., ONNX).

        Returns:
            The path to the exported model file.

        Raises:
            ValueError: If no results folder or ONNX file is found during the export process.
        """
        yolox_model: torch.nn.Module = self.model_context.loaded_model

        if export_format == YoloXModelVersionConversionTargetFramework.COREML.value:
            if architecture := self.model_version_docker_env_variables.get(
                "architecture", None
            ):
                model = self._convert_yolox_to_coreml(model=yolox_model)
                model_name = (
                    f"{self._sanitize_filename(filename=architecture)}.mlpackage"
                )

                save_path = os.path.join(exported_weights_destination_path, model_name)
                model.save(save_path=save_path)

                return save_path
            else:
                raise KeyError(
                    "The architecture of the Model Version could not be found in the Docker environment variables."
                )

        else:
            raise NotImplementedError(
                f"Exporting to {export_format} is not supported for YoloX models. Supported formats are: {YoloXModelVersionConversionTargetFramework.COREML}"
            )

    def _convert_yolox_to_coreml(self, model: torch.nn.Module) -> ct.models.MLModel:
        """
        Converts a YOLOX model to a CoreML model.

        Args:
            model: The YOLOX model to convert.

        Returns:
            The CoreML model.
        """

        yolox_input_size = (self._get_yolox_input_size(),) * 2
        example_input = torch.rand(1, 3, *yolox_input_size)

        traced_model = self._trace_yolox_model(model=model, input_tensor=example_input)

        return ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_input.shape)],
        )

    def _get_yolox_input_size(self) -> int:
        if "image_size" in self.model_version_base_parameters:
            return self.model_version_base_parameters["image_size"]

        elif "input_size" in self.model_version_base_parameters:
            return self.model_version_base_parameters["input_size"]

        elif "imgsz" in self.model_version_base_parameters:
            return self.model_version_base_parameters["imgsz"]

        else:
            raise KeyError(
                "The input size of the Model Version could not be found in the parameters. "
                "Expected keys: 'image_size', 'input_size', 'imgsz'."
            )

    def _trace_yolox_model(
        self, model: torch.nn.Module, input_tensor: torch.Tensor
    ) -> torch.nn.Module:
        """
        Traces the model with a random input tensor to convert it to a TorchScript module.

        Args:
            model: The model to trace.
            input_tensor: The input data to use for tracing.

        Returns:
            The traced model.
        """

        return torch.jit.trace(model, input_tensor)
