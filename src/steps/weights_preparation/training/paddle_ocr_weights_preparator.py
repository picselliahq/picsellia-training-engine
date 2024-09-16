# type: ignore

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext
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
from src.models.steps.weights_preparation.training.paddle_ocr_model_collection_preparator import (
    PaddleOCRModelCollectionPreparator,
)


@step
def paddle_ocr_model_collection_preparator(
    model_collection: PaddleOCRModelCollection,
    dataset_collection: TrainingDatasetCollection[PaddleOCRDatasetContext],
) -> PaddleOCRModelCollection:
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()
    model_collection_preparator = PaddleOCRModelCollectionPreparator(
        model_collection=model_collection,
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
    )
    model_collection = model_collection_preparator.prepare()
    return model_collection
