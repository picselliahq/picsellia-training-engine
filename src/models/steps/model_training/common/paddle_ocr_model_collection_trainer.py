from typing import Union

from picsellia import Experiment

from src.models.model.paddle_ocr.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from src.models.steps.model_training.common.paddle_ocr_model_context_trainer import (
    PaddleOCRModelContextTrainer,
)


class PaddleOCRModelCollectionTrainer:
    def __init__(
        self, model_collection: PaddleOCRModelCollection, experiment: Experiment
    ):
        self.model_collection = model_collection
        self.experiment = experiment
        self.last_logged_epoch: Union[int, None] = None  # Last epoch that was logged

    def train_model_collection(
        self, bbox_epochs: int, text_epochs: int
    ) -> PaddleOCRModelCollection:
        """
        Trains both the bounding box detection and text recognition models in the model collection.
        """
        if bbox_epochs > 0:
            print("Starting training for bounding box model...")
            model_context_trainer = PaddleOCRModelContextTrainer(
                model_context=self.model_collection.bbox_model,
                experiment=self.experiment,
            )
            model_context_trainer.train_model_context()
        else:
            print("Skipping training for bounding box model...")

        if text_epochs > 0:
            print("Starting training for text recognition model...")
            model_context_trainer = PaddleOCRModelContextTrainer(
                model_context=self.model_collection.text_model,
                experiment=self.experiment,
            )
            model_context_trainer.train_model_context()
        else:
            print("Skipping training for text recognition model...")

        return self.model_collection
