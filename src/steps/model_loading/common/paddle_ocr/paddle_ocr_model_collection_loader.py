# type: ignore

import os

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
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
from src.models.steps.model_loading.paddle_ocr.paddle_ocr_model_collection_loader import (
    paddle_ocr_load_model,
)


@step
def paddle_ocr_model_collection_loader(
    model_collection: PaddleOCRModelCollection,
) -> PaddleOCRModelCollection:
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()
    if (
        model_collection.bbox_model.exported_weights_dir
        and model_collection.text_model.exported_weights_dir
        and os.path.exists(model_collection.bbox_model.exported_weights_dir)
        and os.path.exists(model_collection.text_model.exported_weights_dir)
        and os.path.exists(
            os.path.join(model_collection.text_model.weights_dir, "en_dict.txt")
        )
    ):
        loaded_model = paddle_ocr_load_model(
            bbox_model_path_to_load=model_collection.bbox_model.exported_weights_dir,
            text_model_path_to_load=model_collection.text_model.exported_weights_dir,
            character_dict_path_to_load=os.path.join(
                model_collection.text_model.weights_dir, "en_dict.txt"
            ),
            device=context.hyperparameters.device,
        )
        model_collection.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {model_collection.bbox_model.exported_weights_dir} or {model_collection.text_model.exported_weights_dir}. Cannot load model."
        )
    return model_collection
