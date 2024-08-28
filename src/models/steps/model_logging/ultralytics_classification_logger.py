from src.models.steps.model_logging.base_logger import (
    ClassificationMetricMapping,
    Metric,
    BaseLogger,
)

from picsellia import Experiment


class UltralyticsClassificationMetricMapping(ClassificationMetricMapping):
    def __init__(self):
        super().__init__()
        self.add_metric(
            phase="train",
            metric=Metric(
                standard_name="accuracy", framework_name="metrics/accuracy_top1"
            ),
        )
        self.add_metric(
            phase="train",
            metric=Metric(
                standard_name="accuracy_top5", framework_name="metrics/accuracy_top5"
            ),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="loss", framework_name="train/loss"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="learning_rate", framework_name="lr/pg0"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="learning_rate_pg1", framework_name="lr/pg1"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="learning_rate_pg2", framework_name="lr/pg2"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="epoch_time", framework_name="epoch_time"),
        )

        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="accuracy", framework_name="metrics/accuracy_top1"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="accuracy_top5", framework_name="metrics/accuracy_top5"
            ),
        )
        self.add_metric(
            phase="val", metric=Metric(standard_name="loss", framework_name="val/loss")
        )


class UltralyticsClassificationLogger(BaseLogger):
    def __init__(self, experiment: Experiment):
        """
        Initialize the Ultralytics logger with a specific metric mapping.

        Args:
            experiment (Experiment): The experiment object for logging.
        """
        metric_mapping = UltralyticsClassificationMetricMapping()
        super().__init__(experiment=experiment, metric_mapping=metric_mapping)
