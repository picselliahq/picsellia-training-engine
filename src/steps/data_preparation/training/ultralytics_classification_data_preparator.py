import os
import shutil

from src import step
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.steps.data_preparation.common.classification_dataset_context_preparator import (
    ClassificationDatasetContextPreparator,
)


@step
def ultralytics_classification_data_preparator(
    dataset_collection: TrainingDatasetCollection,
) -> TrainingDatasetCollection:
    for dataset_context in dataset_collection:
        organizer = ClassificationDatasetContextPreparator(
            dataset_context=dataset_context,
            dataset_path=dataset_collection.dataset_path,
        )
        new_dataset_context = organizer.organize()
        if new_dataset_context.annotations_dir:
            if os.path.exists(new_dataset_context.annotations_dir):
                shutil.rmtree(new_dataset_context.annotations_dir)
                # new_dataset_context.annotations_dir = None
        dataset_collection[new_dataset_context.dataset_name] = new_dataset_context

    return dataset_collection
