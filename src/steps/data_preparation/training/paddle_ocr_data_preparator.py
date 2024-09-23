import os

from src import step, Pipeline
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext
from src.models.steps.data_preparation.training.paddle_ocr_dataset_context_preparator import (
    PaddleOCRDatasetContextPreparator,
)


@step
def paddle_ocr_dataset_collection_preparator(
    dataset_collection: DatasetCollection[DatasetContext],
) -> DatasetCollection[PaddleOCRDatasetContext]:
    context = Pipeline.get_active_context()
    paddleocr_dataset_collection = DatasetCollection(
        [
            PaddleOCRDatasetContextPreparator(
                dataset_collection["train"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["train"].dataset_name,
                    )
                ),
            ).organize(),
            PaddleOCRDatasetContextPreparator(
                dataset_collection["val"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["val"].dataset_name,
                    )
                ),
            ).organize(),
            PaddleOCRDatasetContextPreparator(
                dataset_collection["test"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["test"].dataset_name,
                    )
                ),
            ).organize(),
        ]
    )
    return paddleocr_dataset_collection
