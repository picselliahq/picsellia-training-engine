from abc import abstractmethod
from typing import TypeVar, Generic

from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.model.model_context import ModelContext
from src.models.steps.model_inferencing.base_model_inference import BaseModelInference

TModelContext = TypeVar("TModelContext", bound=ModelContext)


class ModelContextInference(BaseModelInference, Generic[TModelContext]):
    def __init__(self, model_context: TModelContext):
        self.model_context: TModelContext = model_context

    @abstractmethod
    def predict_on_dataset_context(self, dataset_context: TDatasetContext):
        pass
