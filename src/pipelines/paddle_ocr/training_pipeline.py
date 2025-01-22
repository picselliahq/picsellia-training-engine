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
from src.steps.data_extraction.training.coco_data_extractor import (
    get_coco_dataset_collection,
)
from src.steps.data_preparation.training.paddle_ocr_data_preparator import (
    prepare_paddle_ocr_dataset_collection,
)
from src.steps.model_evaluation.common.paddle_ocr_model_evaluator import (
    evaluate_paddle_ocr_model_collection,
)
from src.steps.model_export.common.paddle_ocr_model_exporter import (
    export_paddle_ocr_model_collection,
)
from src.steps.model_loading.common.paddle_ocr.paddle_ocr_model_collection_loader import (
    load_paddle_ocr_model_collection,
)
from src.steps.model_training.paddle_ocr_trainer import (
    train_paddle_ocr_model_collection,
)
from src.steps.weights_extraction.training.paddle_ocr_weights_extractor import (
    get_paddle_ocr_model_collection,
)
from src.steps.weights_preparation.training.paddle_ocr_weights_preparator import (
    prepare_paddle_ocr_model_collection,
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
    dataset_collection = get_coco_dataset_collection()
    dataset_collection = prepare_paddle_ocr_dataset_collection(
        dataset_collection=dataset_collection
    )
    model_collection = get_paddle_ocr_model_collection()
    model_collection = prepare_paddle_ocr_model_collection(
        model_collection=model_collection, dataset_collection=dataset_collection
    )
    model_collection = train_paddle_ocr_model_collection(
        model_collection=model_collection
    )
    model_collection = export_paddle_ocr_model_collection(
        model_collection=model_collection
    )
    model_collection = load_paddle_ocr_model_collection(
        model_collection=model_collection
    )
    evaluate_paddle_ocr_model_collection(
        model_collection=model_collection, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    paddle_ocr_training_pipeline()
