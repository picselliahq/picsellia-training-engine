import logging
from typing import TypeVar

from picsellia.types.schemas import LogDataType  # type: ignore

from src.picsellia_cv_engine.models.parameters.parameters import Parameters

logger = logging.getLogger("picsellia-engine")


class AugmentationParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)


TAugmentationParameters = TypeVar(
    "TAugmentationParameters", bound=AugmentationParameters
)
