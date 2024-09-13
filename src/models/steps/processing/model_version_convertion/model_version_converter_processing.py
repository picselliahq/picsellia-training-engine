import os
from enum import Enum
from typing import List

import coremltools as ct
import torch.nn

from src.models.model.model_context import ModelContext


class ModelVersionConversionTargetFramework(Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    COREML = "coreml"


class YoloXModelVersionConverterProcessing:
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.
    """

    def __init__(self, model_context: ModelContext):
        self.model_context = model_context
        self.model_version_parameters = model_context.model_version.sync()[
            "base_parameters"
        ]

    def process(
        self,
        target_frameworks: List[ModelVersionConversionTargetFramework],
        output_path: str,
    ) -> None:
        yolox_model: torch.nn.Module = self.model_context.loaded_model

        for target_framework in target_frameworks:
            if target_framework == ModelVersionConversionTargetFramework.COREML:
                model = self._convert_yolox_to_coreml(model=yolox_model)
                model_name = "yoloxs.mlpackage"

            else:
                raise NotImplementedError(
                    f"Target framework '{target_framework.value}' is not supported for conversion."
                )

            # TODO add the exported folder instead
            output_path = os.path.join(
                os.getcwd(), self.model_context.weights_dir, model_name
            )
            model.save(output_path)

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
        if "image_size" in self.model_version_parameters:
            return self.model_version_parameters["image_size"]

        elif "input_size" in self.model_version_parameters:
            return self.model_version_parameters["input_size"]

        elif "imgsz" in self.model_version_parameters:
            return self.model_version_parameters["imgsz"]

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
