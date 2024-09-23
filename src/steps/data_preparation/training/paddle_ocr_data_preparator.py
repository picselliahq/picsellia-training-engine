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
    """
    Prepares and organizes a dataset collection for PaddleOCR training.

    This function takes an existing `DatasetCollection` containing the 'train', 'val', and 'test' dataset contexts,
    and organizes them into a format suitable for PaddleOCR training. It uses the `PaddleOCRDatasetContextPreparator`
    to organize the datasets (e.g., creating necessary directories and moving images) for each dataset split (train, val, test).
    The organized datasets are then stored in a new `DatasetCollection` with `PaddleOCRDatasetContext` types.

    Args:
        dataset_collection (DatasetCollection[DatasetContext]): The original dataset collection containing 'train', 'val', and 'test' splits.

    Returns:
        DatasetCollection[PaddleOCRDatasetContext]: A new dataset collection where each dataset is organized for PaddleOCR,
        with directories properly set up for training, validation, and testing.
    """
    context = Pipeline.get_active_context()

    paddleocr_dataset_collection = DatasetCollection(
        [
            PaddleOCRDatasetContextPreparator(
                dataset_context=dataset_collection["train"],
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
                dataset_context=dataset_collection["val"],
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
                dataset_context=dataset_collection["test"],
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
