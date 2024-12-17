from enum import Enum
from typing import Dict, Optional, Tuple

import torch.nn as nn
from picsellia import Label

from src.models.steps.model_loading.architecture.yolox import (
    YOLOPAFPN,
    YOLOX,
    YOLOXHead,
)


class YoloxArchitecture(Enum):
    YoloXNano = "yolox-nano"
    YoloXTiny = "yolox-tiny"
    YoloXS = "yolox-s"
    YoloXM = "yolox-m"
    YoloXL = "yolox-l"
    YoloXX = "yolox-x"


class YoloX:
    def __init__(self, architecture: str, labelmap: Optional[Dict[str, Label]]) -> None:
        self.architecture = architecture

        self.width, self.depth = self._get_width_and_depth_from_architecture(
            YoloxArchitecture(architecture)
        )
        self.num_classes = len(labelmap) if labelmap is not None else 0
        self.activation_function = "silu"

        self._model = None

    def get_model(self) -> nn.Module:
        if self._model is None:
            self._model = self._build_model()

        return self._model

    def _build_model(self) -> nn.Module:
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]

        backbone = YOLOPAFPN(
            self.depth,
            self.width,
            in_channels=in_channels,
            act=self.activation_function,
        )
        head = YOLOXHead(
            self.num_classes,
            self.width,
            in_channels=in_channels,
            act=self.activation_function,
        )
        model = YOLOX(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)

        return model

    def _get_width_and_depth_from_architecture(
        self, architecture: YoloxArchitecture
    ) -> Tuple[float, float]:
        """
        Get the width and depth factors for a given architecture.

        Args:
            architecture: The architecture to get the factors for.

        Returns:
            Tuple[float, float]: The width and depth factors.
        """
        if architecture == YoloxArchitecture.YoloXNano:
            return 0.25, 0.33
        elif architecture == YoloxArchitecture.YoloXTiny:
            return 0.375, 0.33
        elif architecture == YoloxArchitecture.YoloXS:
            return 0.50, 0.33
        elif architecture == YoloxArchitecture.YoloXM:
            return 0.75, 0.67
        elif architecture == YoloxArchitecture.YoloXL:
            return 1.00, 1.00
        elif architecture == YoloxArchitecture.YoloXX:
            return 1.25, 1.33
        else:
            raise ValueError(
                f"The specified architecture {architecture} is not supported. "
                f"Currently supported architectures are {YoloxArchitecture.__members__.keys()}."
            )
