from typing import List

from src.models.parameters.common.parameters import Parameters


class ProcessingDatalakeAutotaggingParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.tags_list = self.extract_parameter(
            keys=["tags_list"], expected_type=List[str]
        )
        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cuda:0"
        )
        self.batch_size = self.extract_parameter(
            keys=["batch_size"], expected_type=int, default=8
        )
