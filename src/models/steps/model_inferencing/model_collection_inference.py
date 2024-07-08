from abc import abstractmethod
from typing import TypeVar

from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.model.model_collection import ModelCollection

TModelCollection = TypeVar("TModelCollection", bound=ModelCollection)


class ModelCollectionInference:
    def __init__(self, model_collection: TModelCollection):
        self.model_collection = model_collection

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict_on_dataset_context(self, dataset_context: TDatasetContext):
        pass
