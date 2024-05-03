from src.models.parameters.common.parameters import Parameters


class ProcessingDiversifiedDataExtractorParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.datalake = self.extract_parameter(
            keys=["distance", "dist"],
            expected_type=int,
            default=5,
            range_value=(
                0.0,
                float("inf"),
            ),
        )

        self.destination_dataset_name = self.extract_parameter(
            keys=["dataset_name", "dataset"],
            expected_type=str,
            default=log_data["dataset"],
        )

        self.destination_dataset_version_name = self.extract_parameter(
            keys=[
                "destination_dataset_version_name",
                "destination_dataset_version",
                "destination",
            ],
            expected_type=str,
        )
