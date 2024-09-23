# type: ignore

import os

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext
from src.models.model.paddle_ocr.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.paddle_ocr.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from src.models.parameters.training.paddle_ocr.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)


@step
def paddle_ocr_model_collection_extractor() -> PaddleOCRModelCollection:
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_version = context.experiment.get_base_model_version()

    bbox_model = ModelContext(
        model_name="bbox-model",
        model_version=model_version,
        pretrained_weights_name="bbox-pretrained-model",
        trained_weights_name=None,
        config_name="bbox-config",
        exported_weights_name=None,
    )
    text_model = ModelContext(
        model_name="text-model",
        model_version=model_version,
        pretrained_weights_name="text-pretrained-model",
        trained_weights_name=None,
        config_name="text-config",
        exported_weights_name=None,
    )

    model_collection = PaddleOCRModelCollection(
        bbox_model=bbox_model, text_model=text_model
    )
    model_collection.download_weights(
        destination_path=os.path.join(os.getcwd(), context.experiment.name, "model")
    )

    return model_collection
