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
from src.models.steps.model_loading.common.paddle_ocr.paddle_ocr_model_collection_loader import (
    paddle_ocr_load_model,
)


@step
def paddle_ocr_model_collection_loader(
    model_collection: PaddleOCRModelCollection,
) -> PaddleOCRModelCollection:
    """
    Loads a PaddleOCR model collection from pretrained weights if available.

    This function retrieves the active training context and attempts to load the PaddleOCR model collection
    (both bounding box and text recognition models) from their respective pretrained weights directories.
    The function checks for the existence of the required weight files and the character dictionary. If all files
    are present, the models are loaded onto the specified device. If any required files are missing,
    a `FileNotFoundError` is raised.

    Args:
        model_collection (PaddleOCRModelCollection): The PaddleOCR model collection to load pretrained weights into.

    Returns:
        PaddleOCRModelCollection: The model collection with the loaded models.

    Raises:
        FileNotFoundError: If any of the required model weight files or the character dictionary file are not found.
    """
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
