import logging
from picsellia.types.schemas import LogDataType  # type: ignore
from src.models.parameters.parameters import Parameters

logger = logging.getLogger("picsellia")


class HyperParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        self.epochs = self.extract_parameter(
            keys=["epoch", "epochs"], expected_type=int
        )
        self.batch_size = self.extract_parameter(
            keys=["batch_size", "batch"],
            expected_type=int,
            default=8,
        )
        self.image_size = self.extract_parameter(
            keys=["image_size", "imgsz"], expected_type=int
        )
        self.seed = self.extract_parameter(keys=["seed"], expected_type=int, default=0)

        self.validate = self.extract_parameter(
            keys=["validate", "val", "validation"],
            expected_type=bool,
            default=False,
        )


class UltralyticsHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cpu"
        )
        self.use_cache = self.extract_parameter(
            keys=["cache", "use_cache"],
            expected_type=bool,
            default="False",
        )
        self.save_period = self.extract_parameter(
            keys=["save_period"],
            expected_type=int,
            default=-1,
        )
