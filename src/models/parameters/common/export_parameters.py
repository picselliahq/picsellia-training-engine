import logging
from typing import TypeVar

from picsellia.types.schemas import LogDataType  # type: ignore
from src.models.parameters.common.parameters import Parameters

logger = logging.getLogger("picsellia-engine")


class ExportParameters(Parameters):
    """
    Handles the export parameters for model exportation processes.

    Inherits from the base `Parameters` class and is responsible for extracting and managing the
    format in which the model will be exported, such as ONNX or other supported formats.

    Attributes:
        export_format (str): The format in which the model will be exported. Defaults to 'onnx'.
    """

    def __init__(self, log_data: LogDataType):
        """
        Initializes the ExportParameters object by extracting the export format from the provided log data.

        Args:
            log_data (LogDataType): The log data schema that contains the parameters for export.
        """
        super().__init__(log_data=log_data)

        self.export_format = self.extract_parameter(
            keys=["export_format"], expected_type=str, default="onnx"
        )


TExportParameters = TypeVar("TExportParameters", bound=ExportParameters)
