import os

from src import step
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.steps.data_preparation.common.classification_dataset_context_preparator import (
    ClassificationDatasetContextPreparator,
)


@step
def ultralytics_classification_dataset_collection_preparator(
    dataset_collection: TrainingDatasetCollection,
) -> TrainingDatasetCollection:
    for dataset_context in dataset_collection:
        destination_image_dir = str(
            os.path.join(dataset_context.destination_path, dataset_context.dataset_name)
        )
        preparator = ClassificationDatasetContextPreparator(
            dataset_context=dataset_context,
            destination_image_dir=destination_image_dir,
        )
        prepared_dataset_context = preparator.organize()

        dataset_collection[
            prepared_dataset_context.dataset_name
        ] = prepared_dataset_context

    return dataset_collection
