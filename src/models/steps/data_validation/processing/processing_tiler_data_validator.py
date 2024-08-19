from picsellia import Client


class ProcessingTilerDataValidator:
    def __init__(
        self,
        client: Client,
        tile_height: int,
        tile_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        min_area_ratio: float,
        datalake: str,
    ):
        self.client = client
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.min_area_ratio = min_area_ratio
        self.datalake = datalake

    def _validate_tile_size(self) -> None:
        """
        Validate that the slice size is valid.

        Raises:
            ValueError: If the slice size is not valid.
        """
        if self.tile_height <= 0 or self.tile_width <= 0:
            raise ValueError("Slice size must be greater than 0")

    def _validate_ratios(self) -> None:
        """
        Validate that the overlap and min area ratios are within the acceptable range.

        Raises:
            ValueError: If any of the ratios are out of the acceptable range.
        """
        if not (0 <= self.overlap_height_ratio < 1):
            raise ValueError("overlap_height_ratio must be between 0 and 0.99")
        if not (0 <= self.overlap_width_ratio < 1):
            raise ValueError("overlap_width_ratio must be between 0 and 0.99")
        if not (0 <= self.min_area_ratio < 1):
            raise ValueError("min_area_ratio must be between 0 and 0.99")

    def _validate_datalake(self) -> None:
        """
        Validate that the datalake is valid.

        Raises:
            ValueError: If the datalake is not valid.
        """
        datalakes_name = [datalake.name for datalake in self.client.list_datalakes()]
        if self.datalake not in datalakes_name:
            raise ValueError(
                f"Datalake {self.datalake} is not valid, available datalakes are {datalakes_name}"
            )

    def validate(self) -> None:
        """
        Validate the tiler parameters.
        """
        self._validate_tile_size()
        self._validate_ratios()
        self._validate_datalake()
