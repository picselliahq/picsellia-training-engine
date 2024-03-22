from src.models.dataset.dataset_type import DatasetType
from src.steps.data_extraction.utils.dataset_context import DatasetContext


class DatasetCollection:
    def __init__(
        self,
        train_context: DatasetContext,
        val_context: DatasetContext,
        test_context: DatasetContext,
    ):
        self.train = train_context
        self.val = val_context
        self.test = test_context

    def download(self):
        for context_name in [
            DatasetType.TRAIN.value,
            DatasetType.VAL.value,
            DatasetType.TEST.value,
        ]:
            context = getattr(self, context_name)
            if context is not None:
                context.download_assets()
                context.download_coco_file()
