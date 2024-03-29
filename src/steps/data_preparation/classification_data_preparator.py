from src import step
from src.steps.data_preparation.utils.classification_dataset_organizer import (
    ClassificationDatasetOrganizer,
)
from src.models.dataset.dataset_split_name import DatasetSplitName
from src.models.dataset.dataset_collection import DatasetCollection


@step
def classification_data_preparator(
    dataset_collection: DatasetCollection,
) -> DatasetCollection:
    """
    Prepares the dataset collection for classification tasks by organizing images into category directories.

    This function iterates through each dataset context within the provided dataset collection. For dataset
    contexts corresponding to training, validation, and testing splits, it utilizes the ClassificationDatasetOrganizer
    to organize images into directories based on their classification categories. This organization is crucial for
    many machine learning frameworks and simplifies the process of dataset loading for classification models.

    The organization process involves creating a directory for each category and moving images into their
    respective category directories within the dataset's extraction path.

    Args:
        dataset_collection (DatasetCollection): A collection of dataset contexts that include the training,
        validation, and testing splits of the dataset.

    Returns:
        DatasetCollection: The original dataset collection with its dataset contexts now organized by classification
        categories. This modified collection is ready for use in training classification models.

    Note:
        The function modifies the dataset contexts in place by organizing the images into category directories.
        It returns the original dataset collection object, which now references the organized dataset contexts.
    """
    for dataset_context in dataset_collection:
        if dataset_context.dataset_name in [
            DatasetSplitName.TRAIN.value,
            DatasetSplitName.VAL.value,
            DatasetSplitName.TEST.value,
        ]:
            organizer = ClassificationDatasetOrganizer(dataset_context=dataset_context)
            organizer.organize()
    return dataset_collection
