from abc import abstractmethod, ABC

from src.models.dataset.training.training_dataset_collection import TDatasetContext


class BaseModelInference(ABC):
    @abstractmethod
    def predict_on_dataset_context(self, dataset_context: TDatasetContext):
        pass
