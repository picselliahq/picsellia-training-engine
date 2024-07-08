from abc import abstractmethod, ABC
from typing import TypeVar, Generic

from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.model.model_collection import ModelCollection

TModelCollection = TypeVar("TModelCollection", bound=ModelCollection)


class ModelCollectionInference(ABC, Generic[TModelCollection]):
    def __init__(self, model_collection: TModelCollection):
        self.model_collection: TModelCollection = model_collection

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict_on_dataset_context(self, dataset_context: TDatasetContext):
        pass
