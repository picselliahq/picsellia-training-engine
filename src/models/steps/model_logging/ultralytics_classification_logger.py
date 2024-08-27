from src.models.steps.model_logging.classification_logger import ClassificationLogger


class UltralyticsClassificationLogger(ClassificationLogger):
    def _get_default_metric_mappings(self) -> dict:
        # Map Ultralytics-specific metric names to your standard names
        return {
            "train": {
                "metrics/accuracy_top1": "accuracy",
                "metrics/accuracy_top5": "accuracy_top5",
                "train/loss": "loss",
                "lr/pg0": "learning_rate",
                "lr/pg1": "learning_rate_pg1",
                "lr/pg2": "learning_rate_pg2",
                "epoch_time": "epoch_time",
            },
            "val": {
                "val/loss": "loss",
                "metrics/accuracy_top1": "accuracy",
                "metrics/accuracy_top5": "accuracy_top5",
            },
            "test": {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1_score": "f1_score",
                "loss": "loss",
                "confusion_matrix": "confusion_matrix",
            },
        }
