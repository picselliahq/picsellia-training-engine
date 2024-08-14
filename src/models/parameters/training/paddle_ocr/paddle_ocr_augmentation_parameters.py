from src.models.parameters.common.augmentation_parameters import AugmentationParameters
from picsellia.types.schemas import LogDataType


class PaddleOCRAugmentationParameters(AugmentationParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
