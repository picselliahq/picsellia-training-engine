from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaTrainingContext
from src.models.dataset.dataset_collection import DatasetCollection
from src.steps.data_extraction.utils.dataset_handler import DatasetHandler


@step
def data_extractor() -> DatasetCollection:
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
    dataset_handler = DatasetHandler(
        experiment=context.experiment,
        prop_train_split=context.hyperparameters.prop_train_split,
    )
    dataset_collection = dataset_handler.get_dataset_collection()
    dataset_collection.download()
    return dataset_collection
