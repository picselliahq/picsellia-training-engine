from src.models.parameters.common.parameters import Parameters


class ProcessingTilerParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.datalake = self.extract_parameter(
            keys=["datalake"], expected_type=str, default="default"
        )
        self.images_tag = self.extract_parameter(keys=["images_tag"], expected_type=str)
        self.tile_height = self.extract_parameter(
            keys=["tile_height"], expected_type=int
        )
        self.tile_width = self.extract_parameter(keys=["tile_width"], expected_type=int)
        self.overlap_height_ratio = self.extract_parameter(
            keys=["overlap_height_ratio"], expected_type=float, default=0.1
        )
        self.overlap_width_ratio = self.extract_parameter(
            keys=["overlap_width_ratio"], expected_type=float, default=0.1
        )
        self.min_area_ratio = self.extract_parameter(
            keys=["min_area_ratio"], expected_type=float, default=0.1
        )
