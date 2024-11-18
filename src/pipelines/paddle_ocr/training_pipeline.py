# type: ignore

from src import pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.paddle_ocr.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from src.models.parameters.training.paddle_ocr.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from src.steps.data_extraction.training.training_data_extractor import (
    training_dataset_collection_extractor,
)
from src.steps.data_preparation.training.paddle_ocr_data_preparator import (
    paddle_ocr_dataset_collection_preparator,
)
from src.steps.model_evaluation.common.paddle_ocr_model_evaluator import (
    paddle_ocr_model_collection_evaluator,
)
from src.steps.model_export.common.paddle_ocr_model_exporter import (
    paddle_ocr_model_collection_exporter,
)
from src.steps.model_loading.common.paddle_ocr.paddle_ocr_model_collection_loader import (
    paddle_ocr_model_collection_loader,
)
from src.steps.model_training.paddle_ocr_trainer import (
    paddle_ocr_model_collection_trainer,
)
from src.steps.weights_extraction.training.paddle_ocr_weights_extractor import (
    paddle_ocr_model_collection_extractor,
)
from src.steps.weights_preparation.training.paddle_ocr_weights_preparator import (
    paddle_ocr_model_collection_preparator,
)


def get_context() -> (
    PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ]
):
    return PicselliaTrainingContext(
        hyperparameters_cls=PaddleOCRHyperParameters,
        augmentation_parameters_cls=PaddleOCRAugmentationParameters,
        export_parameters_cls=ExportParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def paddle_ocr_training_pipeline():
    dataset_collection = training_dataset_collection_extractor()
    dataset_collection = paddle_ocr_dataset_collection_preparator(
        dataset_collection=dataset_collection
    )
    model_collection = paddle_ocr_model_collection_extractor()
    model_collection = paddle_ocr_model_collection_preparator(
        model_collection=model_collection, dataset_collection=dataset_collection
    )
    model_collection = paddle_ocr_model_collection_trainer(
        model_collection=model_collection
    )
    model_collection = paddle_ocr_model_collection_exporter(
        model_collection=model_collection
    )
    model_collection = paddle_ocr_model_collection_loader(
        model_collection=model_collection
    )
    paddle_ocr_model_collection_evaluator(
        model_collection=model_collection, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    paddle_ocr_training_pipeline()
