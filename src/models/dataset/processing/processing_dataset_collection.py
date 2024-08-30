from typing import Generic

from src.models.dataset.common.dataset_context import TDatasetContext


class ProcessingDatasetCollection(Generic[TDatasetContext]):
    def __init__(
        self,
        input_dataset_context: TDatasetContext,
        output_dataset_context: TDatasetContext,
    ):
        self.input = input_dataset_context
        self.output = output_dataset_context

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter([self.input, self.output])

    def download_assets(self):
        for dataset_context in self:
            dataset_context.download_assets()
