from picsellia import Client


class ProcessingTilerDataValidator:
    def __init__(
        self,
        client: Client,
        tile_height: int,
        tile_width: int,
        datalake: str,
    ):
        self.client = client
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.datalake = datalake

    def _validate_tile_size(self) -> None:
        """
        Validate that the slice size is valid.

        Raises:
            ValueError: If the slice size is not valid.
        """
        if self.tile_height <= 0 or self.tile_width <= 0:
            raise ValueError("Slice size must be greater than 0")

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
        self._validate_tile_size()
        self._validate_datalake()
