from abc import ABC
from typing import Generic

from src.models.model.common.model_collection import TModelCollection


class ModelCollectionPredictor(ABC, Generic[TModelCollection]):
    def __init__(self, model_collection: TModelCollection):
        self.model_collection: TModelCollection = model_collection

        if not hasattr(self.model_collection, "loaded_model"):
            raise ValueError(
                "The model collection does not have a loaded model attribute."
            )
