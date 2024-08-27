from src import step, Pipeline
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext
from src.models.model.paddle_ocr.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from src.models.steps.weights_preparation.training.paddle_ocr_model_collection_preparator import (
    PaddleOCRModelCollectionPreparator,
)


@step
def paddle_ocr_weights_preparator(
    model_collection: PaddleOCRModelCollection,
    dataset_collection: TrainingDatasetCollection[PaddleOCRDatasetContext],
) -> PaddleOCRModelCollection:
    context = Pipeline.get_active_context()
    model_collection_preparator = PaddleOCRModelCollectionPreparator(
        model_collection=model_collection,
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
    )
    model_collection = model_collection_preparator.prepare()
    return model_collection
