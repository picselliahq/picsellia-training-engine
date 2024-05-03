from src.models.parameters.common.parameters import Parameters


class ProcessingBoundingBoxCropperParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.datalake = self.extract_parameter(
            keys=["datalake"], expected_type=str, default="default"
        )
        self.label_name_to_extract = self.extract_parameter(
            keys=["label_name_to_extract"], expected_type=str
        )
