# type: ignore

from src import pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.parameters.common.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.common.hyper_parameters import UltralyticsHyperParameters
from src.steps.data_extraction.training.training_data_extractor import (
    training_data_extractor,
)
from src.steps.data_preparation.training.ultralytics_classification_data_preparator import (
    ultralytics_classification_data_preparator,
)
from src.steps.data_validation.common.classification_data_validator import (
    classification_data_validator,
)
from src.steps.model_evaluation.ultralytics_model_evaluator import (
    ultralytics_model_evaluator,
)
from src.steps.model_export.ultralytics_model_exporter import ultralytics_model_exporter
from src.steps.model_training.ultralytics_trainer import ultralytics_trainer
from src.steps.weights_extraction.training.ultralytics_weights_extractor import (
    ultralytics_weights_extractor,
)


def get_context() -> (
    PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters
    ]
):
    return PicselliaTrainingContext(
        hyperparameters_cls=UltralyticsHyperParameters,
        augmentation_parameters_cls=UltralyticsAugmentationParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_classification_training_pipeline():
    dataset_collection = training_data_extractor(random_seed=2)
    dataset_collection = ultralytics_classification_data_preparator(
        dataset_collection=dataset_collection
    )
    classification_data_validator(dataset_collection=dataset_collection)
    model_context = ultralytics_weights_extractor()
    model_context = ultralytics_trainer(
        model_context=model_context, dataset_collection=dataset_collection
    )
    model_context = ultralytics_model_exporter(model_context)
    ultralytics_model_evaluator(
        model_context=model_context, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    yolov8_classification_training_pipeline()
