import logging

import yaml


from src.models.model.common.model_context import ModelContext
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
        self.bbox_config = self.get_config(self.model_collection.bbox_model)
        self.text_config = self.get_config(self.model_collection.text_model)
        self.bbox_model_context_exporter = PaddleOCRModelContextExporter(
            model_context=self.model_collection.bbox_model, experiment=self.experiment
        )
        self.text_model_context_exporter = PaddleOCRModelContextExporter(
            model_context=self.model_collection.text_model, experiment=self.experiment
        )

    def get_config(self, model_context: ModelContext) -> dict:
        if not model_context.config_path:
            raise ValueError("No configuration file path found in model context")
        with open(model_context.config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

    def export_model_collection(self) -> PaddleOCRModelCollection:
        """
        Exports the trained models in the model collection.
        """
        logger.info("Exporting bounding box model...")
        self.bbox_model_context_exporter.export_model_context(
            exported_model_destination_path=self.bbox_config["Global"][
                "save_model_dir"
            ],
            export_format="paddle",
        )

        logger.info("Exporting text recognition model...")
        self.text_model_context_exporter.export_model_context(
            exported_model_destination_path=self.text_config["Global"][
                "save_model_dir"
            ],
            export_format="paddle",
        )
        return self.model_collection

    def save_model_collection(self) -> None:
        self.bbox_model_context_exporter.save_model_to_experiment(
            exported_weights_dir=self.bbox_config["Global"]["save_model_dir"],
            exported_weights_name="bbox-model-latest",
        )
        self.text_model_context_exporter.save_model_to_experiment(
            exported_weights_dir=self.text_config["Global"]["save_model_dir"],
            exported_weights_name="text-model-latest",
        )
