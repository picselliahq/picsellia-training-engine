import logging

from src.models.model.paddle_ocr.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from src.models.steps.model_export.training.paddle_ocr_model_context_exporter import (
    PaddleOCRModelContextExporter,
)

from picsellia import Experiment

logger = logging.getLogger(__name__)


class PaddleOCRModelCollectionExporter:
    """
    Handles the export of a collection of PaddleOCR models (bounding box and text recognition models).

    This class exports the trained models in the provided PaddleOCRModelCollection, saving them
    in the specified format and to the Picsellia experiment.

    Attributes:
        model_collection (PaddleOCRModelCollection): The collection of models to export, containing the bounding box and text models.
        experiment (Experiment): The Picsellia experiment where the models will be saved.
        bbox_model_context_exporter (PaddleOCRModelContextExporter): Exporter for the bounding box model.
        text_model_context_exporter (PaddleOCRModelContextExporter): Exporter for the text recognition model.
    """

    def __init__(
        self, model_collection: PaddleOCRModelCollection, experiment: Experiment
    ):
        """
        Initializes the PaddleOCRModelCollectionExporter with the given model collection and experiment.

        Args:
            model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models to export.
            experiment (Experiment): The Picsellia experiment where the models will be saved.
        """
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
        Exports the trained models in the model collection to the specified format.

        This method handles the export process for both the bounding box and text recognition models.
        If the models' export directories are not found, an error is raised. After successful export,
        the updated model collection is returned.

        Args:
            export_format (str): The format in which the models will be exported (e.g., 'onnx', 'tensorflow').

        Returns:
            PaddleOCRModelCollection: The updated model collection with exported models.

        Raises:
            ValueError: If the exported weights directory is not found in the model collection.
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
        """
        Saves the exported models from the model collection to the Picsellia experiment.

        This method uploads the exported bounding box and text recognition models to the associated experiment.
        If the models' export directories are not found, an error is raised.

        Raises:
            ValueError: If the exported weights directory is not found in the model collection.
        """
        if (
            not self.model_collection.bbox_model.exported_weights_dir
            or not self.model_collection.text_model.exported_weights_dir
        ):
            raise ValueError("No exported weights directory found in model collection")

        logger.info("Saving bounding box model to experiment...")
        self.bbox_model_context_exporter.save_model_to_experiment(
            exported_weights_dir=self.model_collection.bbox_model.exported_weights_dir,
            exported_weights_name="bbox-model-latest",
        )

        logger.info("Saving text recognition model to experiment...")
        self.text_model_context_exporter.save_model_to_experiment(
            exported_weights_dir=self.model_collection.text_model.exported_weights_dir,
            exported_weights_name="text-model-latest",
        )
