from src import step
from src.models.dataset.dataset_organizer import ClassificationDatasetOrganizer
from src.steps.data_extraction.utils.dataset_collection import DatasetCollection
from src.models.dataset.dataset_split_name import DatasetSplitName


@step
def classification_data_preparator(dataset_collection: DatasetCollection):
    for context_name in [
        DatasetSplitName.TRAIN.value,
        DatasetSplitName.VAL.value,
        DatasetSplitName.TEST.value,
    ]:
        dataset_context = getattr(dataset_collection, context_name)
        if dataset_context is not None:
            converter = ClassificationDatasetOrganizer(dataset_context=dataset_context)
            organized_dataset_context = converter.organize()
            setattr(dataset_collection, context_name, organized_dataset_context)
    return dataset_collection
