from src import step
from src.enums import DatasetSplitName
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.steps.data_preparation.common.classification_dataset_context_preparator import (
    ClassificationDatasetContextPreparator,
)


@step
def classification_data_preparator(
    dataset_collection: DatasetCollection,
) -> DatasetCollection:
    """
    Example:
        Assume `dataset_collection` comprises unorganized images across training, validation, and testing splits.
        After applying `classification_data_preparator`, the images within each split are reorganized into
        directories named after their classification categories. This reorganization aids in simplifying dataset
        loading and usage for training classification models.

        Before applying `classification_data_preparator`:
        ```
        dataset/
        ├── train/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── image3.jpg
        ├── val/
        │   ├── image4.jpg
        │   ├── image5.jpg
        │   └── image6.jpg
        └── test/
            ├── image7.jpg
            ├── image8.jpg
            └── image9.jpg
        ```

        After applying `classification_data_preparator`:
        ```
        dataset/
        ├── train/
        │   ├── category1/
        │   │   ├── image1.jpg
        │   │   └── image3.jpg
        │   └── category2/
        │       └── image2.jpg
        ├── val/
        │   ├── category1/
        │   │   └── image4.jpg
        │   └── category2/
        │       ├── image5.jpg
        │       └── image6.jpg
        └── test/
            ├── category1/
            │   └── image7.jpg
            └── category2/
                ├── image8.jpg
                └── image9.jpg
        ```
    """
    for dataset_context in dataset_collection:
        if dataset_context.dataset_name in [
            DatasetSplitName.TRAIN.value,
            DatasetSplitName.VAL.value,
            DatasetSplitName.TEST.value,
        ]:
            organizer = ClassificationDatasetContextPreparator(
                dataset_context=dataset_context
            )
            organizer.organize()
    return dataset_collection
