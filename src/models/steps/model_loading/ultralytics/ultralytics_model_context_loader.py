from src.models.model.common.model_context import ModelContext

from ultralytics import YOLO


class UltralyticsModelContextLoader:
    def __init__(self, model_context: ModelContext):
        self.model_context = model_context

    def load_model(self) -> ModelContext:
        """
        Loads the Ultralytics model using the pretrained model path.
        Raises an error if the pretrained model path is not set.
        """
        if self.model_context.pretrained_model_path is None:
            raise ValueError("Pretrained model path is not set")
        loaded_model = YOLO(self.model_context.pretrained_model_path)
        self.model_context.loaded_model = loaded_model
        return self.model_context
