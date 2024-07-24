from src import step
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext
from src.models.steps.data_preparation.training.paddle_ocr_dataset_context_preparator import (
    PaddleOCRDatasetContextPreparator,
)


@step
def paddle_ocr_data_preparator(
    dataset_collection: DatasetCollection[DatasetContext],
) -> DatasetCollection[PaddleOCRDatasetContext]:
    paddleocr_dataset_collection = DatasetCollection(
        train_dataset_context=PaddleOCRDatasetContextPreparator(
            dataset_collection.train
        ).organize(),
        val_dataset_context=PaddleOCRDatasetContextPreparator(
            dataset_collection.val
        ).organize(),
        test_dataset_context=PaddleOCRDatasetContextPreparator(
            dataset_collection.test
        ).organize(),
    )
    return paddleocr_dataset_collection
