from src.models.steps.model_logging.base_logger import BaseLogger, MetricMapping, Metric
from picsellia import Experiment


class ClassificationMetricMapping(MetricMapping):
    """
    Defines the metric mapping for classification tasks across different phases (train, val, test).

    This class adds common metrics such as accuracy, loss, learning rate, and confusion matrix
    for the phases 'train', 'val', and 'test' in a classification model workflow.
    """

    def __init__(self):
        """
        Initializes the classification metric mapping with predefined metrics for each phase.
        """
        super().__init__()
        self.add_metric(phase="train", metric=Metric("accuracy"))
        self.add_metric(phase="train", metric=Metric("loss"))
        self.add_metric(phase="train", metric=Metric("learning_rate"))
        self.add_metric(phase="train", metric=Metric("epoch_time"))

        self.add_metric(phase="val", metric=Metric("accuracy"))
        self.add_metric(phase="val", metric=Metric("loss"))

        self.add_metric(phase="test", metric=Metric("accuracy"))
        self.add_metric(phase="test", metric=Metric("precision"))
        self.add_metric(phase="test", metric=Metric("recall"))
        self.add_metric(phase="test", metric=Metric("f1_score"))
        self.add_metric(phase="test", metric=Metric("loss"))
        self.add_metric(phase="test", metric=Metric("confusion_matrix"))


class ClassificationLogger(BaseLogger):
    """
    Logger for classification tasks, using the predefined metric mapping.

    This class logs metrics like accuracy, loss, precision, and others during training,
    validation, and testing phases of a classification model.
    """

    def __init__(self, experiment: Experiment):
        """
        Initializes the ClassificationLogger with an experiment and a classification metric mapping.

        Args:
            experiment (Experiment): The experiment object for logging classification metrics.
        """
        super().__init__(
            experiment=experiment, metric_mapping=ClassificationMetricMapping()
        )
