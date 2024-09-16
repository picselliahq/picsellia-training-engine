from src.models.dataset.processing.datalake_context import DatalakeContext


class DatalakeCollection:
    def __init__(
        self,
        input_datalake_context: DatalakeContext,
        output_datalake_context: DatalakeContext,
    ):
        self.input = input_datalake_context
        self.output = output_datalake_context

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter([self.input, self.output])

    def download_all(self):
        for datalake_context in self:
            datalake_context.download_data(image_dir=datalake_context.image_dir)
