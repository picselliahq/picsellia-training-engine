from typing import Union

from src.models.parameters.common.parameters import Parameters


class ProcessingTilerParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.datalake = self.extract_parameter(
            keys=["datalake"], expected_type=str, default="default"
        )
        self.data_tag = self.extract_parameter(keys=["data_tag"], expected_type=str)
        self.tile_height = self.extract_parameter(
            keys=["tile_height"], expected_type=int, range_value=(0, float("inf"))
        )
        self.tile_width = self.extract_parameter(
            keys=["tile_width"], expected_type=int, range_value=(0, float("inf"))
        )
        self.overlap_height_ratio = self.extract_parameter(
            keys=["overlap_height_ratio"],
            expected_type=float,
            default=0.1,
            range_value=(0, 0.99),
        )
        self.overlap_width_ratio = self.extract_parameter(
            keys=["overlap_width_ratio"],
            expected_type=float,
            default=0.1,
            range_value=(0, 0.99),
        )
        self.min_annotation_area_ratio = self.extract_parameter(
            keys=["min_annotation_area_ratio", "min_area_ratio"],
            expected_type=float,
            default=0.0,
            range_value=(0, 0.99),
        )
        self.min_annotation_width = self.extract_parameter(
            keys=["min_annotation_width"],
            expected_type=Union[int, None],
            default=30,
            range_value=(0, float("inf")),
        )
        self.min_annotation_height = self.extract_parameter(
            keys=["min_annotation_height"],
            expected_type=Union[int, None],
            default=30,
            range_value=(0, float("inf")),
        )
        self.constant_value = self.extract_parameter(
            keys=["constant_value"],
            expected_type=int,
            default=114,
            range_value=(0, 255),
        )
        self.fix_annotation = self.extract_parameter(
            keys=["fix_annotation"], expected_type=bool, default=True
        )
