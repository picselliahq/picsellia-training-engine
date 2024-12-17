import re
from typing import List

from src.models.parameters.common.parameters import Parameters


class ProcessingDatalakeAutotaggingParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.tags_list: List[str] = re.findall(
            r"'(.*?)'", self.extract_parameter(keys=["tags_list"], expected_type=str)
        )
        self.batch_size = self.extract_parameter(
            keys=["batch_size"], expected_type=int, default=8
        )
