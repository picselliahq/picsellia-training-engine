import os
from src import step, Pipeline
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.steps.data_preparation.common.classification_dataset_context_preparator import (
    ClassificationDatasetContextPreparator,
)


@step
def ultralytics_classification_dataset_collection_preparator(
    dataset_collection: DatasetCollection,
) -> DatasetCollection:
    context = Pipeline.get_active_context()
    for dataset_context in dataset_collection:
        destination_path = str(
            os.path.join(
                os.getcwd(),
                context.experiment.name,
                "ultralytics_dataset",
                dataset_context.dataset_name,
            )
        )
        preparator = ClassificationDatasetContextPreparator(
            dataset_context=dataset_context,
            destination_path=destination_path,
        )
        prepared_dataset_context = preparator.organize()

        dataset_collection[
            prepared_dataset_context.dataset_name
        ] = prepared_dataset_context

    dataset_collection.dataset_path = os.path.join(
        os.getcwd(), context.experiment.name, "ultralytics_dataset"
    )

    return dataset_collection
