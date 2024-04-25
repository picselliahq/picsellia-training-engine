from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import (
    PicselliaTrainingContext,
    PicselliaProcessingContext,
)
from src.models.dataset.dataset_collection import DatasetCollection
from src.models.dataset.dataset_context import DatasetContext
from src.steps.data_extraction.utils.experiment_dataset_collection_extractor import (
    ExperimentDatasetCollectionExtractor,
)
from src.steps.data_extraction.utils.processing_dataset_context_extractor import (
    ProcessingDatasetContextExtractor,
)


@step
def training_data_extractor() -> DatasetCollection:
    """
    Extracts datasets from an experiment and prepares them for training.

    This function retrieves the active training context from the pipeline, uses it to initialize a DatasetHandler
    with the current experiment and the proportion of the training split defined in the hyperparameters. It then
    retrieves a DatasetCollection of datasets ready for use in training, validation, and testing, and downloads
    all necessary assets and annotations.

    The function is designed to be used as a step in a Picsellia Pipeline, making it part of the automated
    data preparation and model training process.

    Returns:
        - DatasetCollection: A collection of dataset contexts prepared for the training, including training,
        validation, and testing splits, with all necessary assets and annotations downloaded.

    Raises:
        ResourceNotFoundError: If any of the expected dataset splits are not found in the experiment.
        RuntimeError: If an invalid number of datasets are attached to the experiment.

    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    dataset_collection_extractor = ExperimentDatasetCollectionExtractor(
        experiment=context.experiment,
        train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
    )
    dataset_collection = dataset_collection_extractor.get_dataset_collection()
    dataset_collection.download_assets()
    return dataset_collection


@step
def processing_data_extractor() -> DatasetContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    dataset_context_extractor = ProcessingDatasetContextExtractor(
        job_id=context.job_id, dataset_version=context.input_dataset_version
    )
    dataset_context = dataset_context_extractor.get_dataset_context()
    dataset_context.download_assets()
    return dataset_context
