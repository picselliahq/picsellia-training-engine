from src import step, Pipeline
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from src.models.steps.weights_preparator.training.paddle_ocr_model_collection_preparator import (
    PaddleOCRModelCollectionPreparator,
)


@step
def paddle_ocr_weights_preparator(
    model_collection: PaddleOCRModelCollection, dataset_collection: DatasetCollection
) -> PaddleOCRModelCollection:
    context = Pipeline.get_active_context()
    model_collection_preparator = PaddleOCRModelCollectionPreparator(
        model_collection=model_collection,
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
    )
    model_collection = model_collection_preparator.prepare()
    return model_collection
