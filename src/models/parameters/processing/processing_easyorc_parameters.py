from src.models.parameters.common.parameters import Parameters


class ProcessingEasyOcrParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.language = self.extract_parameter(
            keys=["language"],
            expected_type=str,
            default="en",
        )
