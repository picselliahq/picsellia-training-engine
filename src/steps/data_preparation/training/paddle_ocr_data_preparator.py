from src import step
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext
from src.models.steps.data_preparation.training.paddle_ocr_dataset_context_preparator import (
    PaddleOCRDatasetContextPreparator,
)


@step
def paddle_ocr_dataset_collection_preparator(
    dataset_collection: TrainingDatasetCollection[DatasetContext],
) -> TrainingDatasetCollection[PaddleOCRDatasetContext]:
    paddleocr_dataset_collection = TrainingDatasetCollection(
        [
            PaddleOCRDatasetContextPreparator(dataset_collection["train"]).organize(),
            PaddleOCRDatasetContextPreparator(dataset_collection["val"]).organize(),
            PaddleOCRDatasetContextPreparator(dataset_collection["test"]).organize(),
        ]
    )
    return paddleocr_dataset_collection
