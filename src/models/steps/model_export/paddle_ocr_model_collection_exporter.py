import logging


from src.models.model.paddle_ocr.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from src.models.steps.model_export.paddle_ocr_model_context_exporter import (
    PaddleOCRModelContextExporter,
)

from picsellia import Experiment

logger = logging.getLogger(__name__)


class PaddleOCRModelCollectionExporter:
    def __init__(
        self, model_collection: PaddleOCRModelCollection, experiment: Experiment
    ):
        self.model_collection = model_collection
        self.experiment = experiment
        self.bbox_model_context_exporter = PaddleOCRModelContextExporter(
            model_context=self.model_collection.bbox_model, experiment=self.experiment
        )
        self.text_model_context_exporter = PaddleOCRModelContextExporter(
            model_context=self.model_collection.text_model, experiment=self.experiment
        )

    def export_model_collection(self, export_format: str) -> PaddleOCRModelCollection:
        """
        Exports the trained models in the model collection.
        """
        if (
            not self.model_collection.bbox_model.exported_weights_dir
            or not self.model_collection.text_model.exported_weights_dir
        ):
            raise ValueError("No exported weights directory found in model collection")

        logger.info("Exporting bounding box model...")
        self.bbox_model_context_exporter.export_model_context(
            exported_model_destination_path=self.model_collection.bbox_model.exported_weights_dir,
            export_format=export_format,
        )

        logger.info("Exporting text recognition model...")
        self.text_model_context_exporter.export_model_context(
            exported_model_destination_path=self.model_collection.text_model.exported_weights_dir,
            export_format=export_format,
        )
        return self.model_collection

    def save_model_collection(self) -> None:
        if (
            not self.model_collection.bbox_model.exported_weights_dir
            or not self.model_collection.text_model.exported_weights_dir
        ):
            raise ValueError("No exported weights directory found in model collection")
        self.bbox_model_context_exporter.save_model_to_experiment(
            exported_weights_dir=self.model_collection.bbox_model.exported_weights_dir,
            exported_weights_name="bbox-model-latest",
        )
        self.text_model_context_exporter.save_model_to_experiment(
            exported_weights_dir=self.model_collection.text_model.exported_weights_dir,
            exported_weights_name="text-model-latest",
        )
