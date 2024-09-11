import os

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.steps.data_extraction.training.training_dataset_collection_extractor import (
    TrainingDatasetCollectionExtractor,
)
from src.steps.data_extraction.utils.image_utils import (
    log_labelmap,
    log_objects_distribution,
)


@step
def training_dataset_collection_extractor() -> TrainingDatasetCollection:
    """
    Extracts datasets from an experiment and prepares them for training.

    This function retrieves the active training context from the pipeline, uses it to initialize a ExperimentDatasetCollectionExtractor
    with the current experiment and the proportion of the training split defined in the hyperparameters. It then
    retrieves a DatasetCollection of datasets ready for use in training, validation, and testing, and downloads
    all necessary assets and annotations.

    The function is designed to be used as a step in a Picsellia Pipeline, making it part of the automated
    data preparation and model training pipeline.

    Returns:
        - TrainingDatasetCollection: A collection of dataset contexts prepared for the training, including training,
        validation, and testing splits, with all necessary assets and annotations downloaded.

    Raises:
        ResourceNotFoundError: If any of the expected dataset splits are not found in the experiment.
        RuntimeError: If an invalid number of datasets are attached to the experiment.

    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    dataset_collection_extractor = TrainingDatasetCollectionExtractor(
        experiment=context.experiment,
        train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
    )
    dataset_collection = dataset_collection_extractor.get_dataset_collection(
        destination_path=os.path.join(os.getcwd(), context.experiment.name, "dataset"),
        random_seed=context.hyperparameters.seed,
    )
    dataset_collection.download_all()

    log_labelmap(
        labelmap=dataset_collection["train"].labelmap,
        experiment=context.experiment,
        log_name="labelmap",
    )
    for dataset_context in dataset_collection:
        log_objects_distribution(
            coco_file=dataset_context.coco_file,
            experiment=context.experiment,
            log_name=f"{dataset_context.dataset_name}/objects_distribution",
        )

    return dataset_collection
