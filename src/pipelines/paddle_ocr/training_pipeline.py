# type: ignore

from src import pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.parameters.common.paddle_ocr.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from src.models.parameters.common.paddle_ocr.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from src.steps.data_extraction.training.training_data_extractor import (
    training_data_extractor,
)
from src.steps.data_preparation.training.paddle_ocr_data_preparator import (
    paddle_ocr_data_preparator,
)
from src.steps.model_training.paddle_ocr_trainer import paddle_ocr_trainer
from src.steps.weights_extraction.training.paddle_ocr_weights_extractor import (
    paddle_ocr_weights_extractor,
)
from src.steps.weights_preparation.training.paddle_ocr_weights_preparator import (
    paddle_ocr_weights_preparator,
)


def get_context() -> (
    PicselliaTrainingContext[PaddleOCRHyperParameters, PaddleOCRAugmentationParameters]
):
    return PicselliaTrainingContext(
        hyperparameters_cls=PaddleOCRHyperParameters,
        augmentation_parameters_cls=PaddleOCRAugmentationParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def paddle_ocr_training_pipeline():
    dataset_collection = training_data_extractor()
    dataset_collection = paddle_ocr_data_preparator(
        dataset_collection=dataset_collection
    )
    model_collection = paddle_ocr_weights_extractor()
    model_collection = paddle_ocr_weights_preparator(
        model_collection=model_collection, dataset_collection=dataset_collection
    )
    model_collection = paddle_ocr_trainer(model_collection)


if __name__ == "__main__":
    paddle_ocr_training_pipeline()
