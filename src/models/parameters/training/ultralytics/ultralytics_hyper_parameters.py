from src.models.parameters.common.hyper_parameters import HyperParameters

from picsellia.types.schemas import LogDataType


class UltralyticsHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cpu"
        )
        self.use_cache = self.extract_parameter(
            keys=["cache", "use_cache"],
            expected_type=bool,
            default=False,
        )
        self.save_period = self.extract_parameter(
            keys=["save_period"],
            expected_type=int,
            default=-1,
        )
