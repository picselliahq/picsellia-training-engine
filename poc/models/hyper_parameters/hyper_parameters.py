class HyperParameters:
    def __init__(
        self, epochs: int, batch_size: int, image_size: int, seed: int, validate: bool
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.seed = seed
        self.validate = validate


class UltralyticsHyperParameters(HyperParameters):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        image_size: int,
        device: str,
        cache_mode: str,
        is_deterministic: bool,
        seed: int,
        save_period: int,
        validate: bool,
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            validate=validate,
        )
        self.device = device
        self.cache_mode = cache_mode
        self.is_deterministic = is_deterministic
        self.save_period = save_period

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the Ultralytics-specific hyperparameters.
        """
        return {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "device": self.device,
            "cache": self.cache_mode,
            "deterministic": self.is_deterministic,
            "seed": self.seed,
            "save_period": self.save_period,
            "val": self.validate,
        }
