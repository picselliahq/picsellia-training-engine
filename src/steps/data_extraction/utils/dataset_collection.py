from src.steps.data_extraction.utils.dataset_context import DatasetContext


class DatasetCollection:
    def __init__(
        self,
        train_dataset_context: DatasetContext,
        val_dataset_context: DatasetContext,
        test_dataset_context: DatasetContext,
    ):
        self.train = train_dataset_context
        self.val = val_dataset_context
        self.test = test_dataset_context

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter([self.train, self.val, self.test])

    def download(self):
        for dataset_context in self:
            dataset_context.download_assets()
            dataset_context.download_coco_file()
