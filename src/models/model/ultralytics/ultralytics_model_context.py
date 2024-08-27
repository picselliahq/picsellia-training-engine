from abc import ABC
from typing import Optional, Dict

from picsellia import ModelVersion, Label
from ultralytics import YOLO

from src.models.model.common.model_context import ModelContext


class UltralyticsModelContext(ModelContext, ABC):
    def __init__(
        self,
        model_name: str,
        model_version: ModelVersion,
        destination_path: str,
        labelmap: Optional[Dict[str, Label]] = None,
        prefix_model_name: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            destination_path=destination_path,
            labelmap=labelmap,
            prefix_model_name=prefix_model_name,
        )

    def load_model(self) -> None:
        if self.pretrained_model_path is None:
            raise ValueError("Pretrained model path is not set")
        self.loaded_model = YOLO(self.pretrained_model_path)
