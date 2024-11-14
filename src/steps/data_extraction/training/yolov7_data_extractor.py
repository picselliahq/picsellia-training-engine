import os

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.yolov7_dataset_collection import Yolov7DatasetCollection
from src.models.steps.data_extraction.training.training_dataset_collection_extractor import (
    TrainingDatasetCollectionExtractor,
)

from src.models.utils.dataset_logging import (
    log_labelmap,
)


@step
def yolov7_dataset_collection_extractor() -> Yolov7DatasetCollection:
    """
    Extracts datasets from an experiment and prepares them for training, validation, and testing.

    This function retrieves the active training context from the pipeline and uses it to initialize a
    `TrainingDatasetCollectionExtractor` with the current experiment and the training split ratio from the
    hyperparameters. It retrieves a `DatasetCollection` of datasets ready for use in training, validation,
    and testing, downloading all necessary assets and annotations.

    The function also logs the labelmap and the objects distribution for each dataset split in the collection,
    facilitating data analysis and tracking in the experiment.

    Returns:
        DatasetCollection: A collection of dataset contexts prepared for training, validation, and testing,
        with all necessary assets and annotations downloaded.

    Raises:
        ResourceNotFoundError: If any of the expected dataset splits (train, validation, test) are not found in the experiment.
        RuntimeError: If an invalid number of datasets are attached to the experiment.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    dataset_collection_extractor = TrainingDatasetCollectionExtractor(
        experiment=context.experiment,
        train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
    )

    dataset_collection = dataset_collection_extractor.get_dataset_collection(
        random_seed=context.hyperparameters.seed,
    )

    yolov7_dataset_collection = Yolov7DatasetCollection(
        datasets=list(dataset_collection.datasets.values())
    )

    yolov7_dataset_collection.download_all(
        destination_path=os.path.join(os.getcwd(), context.experiment.name, "dataset"),
        use_id=True,
    )

    log_labelmap(
        labelmap=dataset_collection["train"].labelmap,
        experiment=context.experiment,
        log_name="labelmap",
    )

    return yolov7_dataset_collection
