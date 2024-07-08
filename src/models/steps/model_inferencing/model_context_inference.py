from abc import abstractmethod
from typing import TypeVar

from src.models.model.model_context import ModelContext

TModelContext = TypeVar("TModelContext", bound=ModelContext)


class ModelContextInference:
    def __init__(self, model_context: TModelContext):
        self.model_context = model_context

    @abstractmethod
    def load_model(self):
        pass

    def infer(self, image):
        pass
