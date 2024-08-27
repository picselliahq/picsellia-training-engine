import os
import shutil
from pathlib import Path

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
    def remove_existing_annotations(annotations_dir: Path) -> None:
        if annotations_dir and os.path.exists(annotations_dir):
            shutil.rmtree(annotations_dir)

    for dataset_context in dataset_collection:
        preparator = ClassificationDatasetContextPreparator(
            dataset_context=dataset_context,
            dataset_path=dataset_collection.dataset_path,
        )

        new_dataset_context = preparator.organize()
        remove_existing_annotations(new_dataset_context.annotations_dir)

        dataset_collection[new_dataset_context.dataset_name] = new_dataset_context

    return dataset_collection
