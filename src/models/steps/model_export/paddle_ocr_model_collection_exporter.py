from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from src.models.steps.model_export.paddle_ocr_model_context_exporter import (
    PaddleOCRModelContextExporter,
)

from picsellia import Experiment


class PaddleOCRModelCollectionExporter:
    def __init__(
        self, model_collection: PaddleOCRModelCollection, experiment: Experiment
    ):
        self.model_collection = model_collection
        self.experiment = experiment

    def export_and_save_model_collection(self):
        """
        Exports the trained models in the model collection.
        """
        print("Exporting bounding box model...")
        model_context_exporter = PaddleOCRModelContextExporter(
            model_context=self.model_collection.bbox_model, experiment=self.experiment
        )
        self.model_collection.bbox_model = (
            model_context_exporter.export_and_save_model_context()
        )

        print("Exporting text recognition model...")
        model_context_exporter = PaddleOCRModelContextExporter(
            model_context=self.model_collection.text_model, experiment=self.experiment
        )
        self.model_collection.text_model = (
            model_context_exporter.export_and_save_model_context()
        )
        return self.model_collection
