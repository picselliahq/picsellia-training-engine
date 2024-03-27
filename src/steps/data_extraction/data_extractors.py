from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaTrainingContext
from src.steps.data_extraction.utils.dataset_collection import DatasetCollection
from src.steps.data_extraction.utils.dataset_handler import DatasetHandler


@step
def data_extractor() -> DatasetCollection:
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    dataset_handler = DatasetHandler(
        experiment=context.experiment,
        prop_train_split=context.hyperparameters.prop_train_split,
    )
    dataset_collection = dataset_handler.get_dataset_collection()
    dataset_collection.download()
    return dataset_collection
