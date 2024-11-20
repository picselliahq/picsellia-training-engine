from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.common.parameters import Parameters
from src.models.steps.model_export.training.yolox_model_context_exporter import (
    YoloXModelVersionConversionTargetFramework,
)


class ProcessingYoloXModelVersionConverterParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cpu"
        )

        self.input_model_file_name = "best-ckpt-60"  # TODO replace
        # self.extract_parameter(
        #    keys=["input_model_file_name"],
        #    expected_type=str,
        # )


class ProcessingYoloXModelVersionConverterExportParameters(ExportParameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.export_format = self.extract_parameter(
            keys=["export_format", "format"],
            expected_type=YoloXModelVersionConversionTargetFramework,
            default=YoloXModelVersionConversionTargetFramework.COREML,
        )

        self.exported_file_name = self.extract_parameter(
            keys=["output_model_file_name"],
            expected_type=str,
            default=f"yolox-{self.export_format.value}",
        )
