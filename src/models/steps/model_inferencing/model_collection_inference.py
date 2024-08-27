from abc import abstractmethod
from typing import TypeVar, Generic

from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.common.model_collection import ModelCollection
from src.models.steps.model_inferencing.base_model_inference import BaseModelInference

TModelCollection = TypeVar("TModelCollection", bound=ModelCollection)


class ModelCollectionInference(BaseModelInference, Generic[TModelCollection]):
    def __init__(self, model_collection: TModelCollection):
        self.model_collection: TModelCollection = model_collection

    @abstractmethod
    def predict_on_dataset_context(self, dataset_context: TDatasetContext):
        pass
