from src.models.steps.model_logging.base_logger import BaseLogger


class ClassificationLogger(BaseLogger):
    def _get_default_metric_mappings(self) -> dict:
        return {
            "train": {
                "accuracy": "accuracy",
                "loss": "loss",
                "learning_rate": "learning_rate",
                "epoch_time": "epoch_time",
            },
            "val": {"accuracy": "accuracy", "loss": "loss"},
            "test": {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1_score": "f1_score",
                "loss": "loss",
                "confusion_matrix": "confusion_matrix",
            },
        }
