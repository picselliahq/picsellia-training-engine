from src.models.parameters.common.parameters import Parameters


class ProcessingYoloXModelVersionConverterParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.distance_threshold = self.extract_parameter(
            keys=["target_framework", "framework"],
            expected_type=str,
        )

        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cpu"
        )

        # TODO remove when processing is integrated
        self.model_version_id = "0191e1a8-94fe-72b8-a407-e63e8503aaf8"

        self.weights_filename = "model-latest"  # self.extract_parameter(
        #     keys=["weights_filename", "weights"],
        #     expected_type=str,
        # )
