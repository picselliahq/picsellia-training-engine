import logging
from typing import TypeVar

from picsellia.types.schemas import LogDataType  # type: ignore

from src.models.parameters.common.parameters import Parameters

logger = logging.getLogger("picsellia-engine")


class ExportParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        self.export_format = self.extract_parameter(
            keys=["export_format"], expected_type=str, default="onnx"
        )


TExportParameters = TypeVar("TExportParameters", bound=ExportParameters)
