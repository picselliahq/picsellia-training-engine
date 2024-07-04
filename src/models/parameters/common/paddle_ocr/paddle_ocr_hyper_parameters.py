from picsellia.types.schemas import LogDataType

from src.models.parameters.common.parameters import Parameters


class PaddleOCRHyperParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.train_set_split_ratio = self.extract_parameter(
            keys=["prop_train_split", "train_set_split_ratio"],
            expected_type=float,
            default=0.8,
        )
        self.bbox_epochs = self.extract_parameter(
            keys=["bbox_epoch"], expected_type=int, default=100
        )
        self.text_epochs = self.extract_parameter(
            keys=["text_epoch"], expected_type=int, default=100
        )
        self.bbox_batch_size = self.extract_parameter(
            keys=["bbox_batch_size"], expected_type=int, default=8
        )
        self.text_batch_size = self.extract_parameter(
            keys=["text_batch_size"], expected_type=int, default=8
        )
        self.bbox_learning_rate = self.extract_parameter(
            keys=["bbox_learning_rate"], expected_type=float, default=0.001
        )
        self.text_learning_rate = self.extract_parameter(
            keys=["text_learning_rate"], expected_type=float, default=0.001
        )
        self.bbox_save_epoch_step = self.extract_parameter(
            keys=["bbox_save_epoch_step"], expected_type=int, default=10
        )
        self.text_save_epoch_step = self.extract_parameter(
            keys=["text_save_epoch_step"], expected_type=int, default=10
        )
        self.max_text_length = self.extract_parameter(
            keys=["max_text_length"], expected_type=int, default=25
        )
