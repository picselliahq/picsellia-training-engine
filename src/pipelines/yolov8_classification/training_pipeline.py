# type: ignore

from src import pipeline
from src.models.contexts.picsellia_context import (
    PicselliaTrainingContext,
)
from src.models.parameters.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.hyper_parameters import UltralyticsHyperParameters
from src.steps.data_extraction.data_extractor import training_data_extractor
from src.steps.data_preparation.classification_data_preparator import (
    classification_data_preparator,
)
from src.steps.data_validation.classification_data_validator import (
    classification_data_validator,
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
    # Data pipeline
    dataset_collection = training_data_extractor()
    dataset_collection = classification_data_preparator(
        dataset_collection=dataset_collection
    )
    classification_data_validator(dataset_collection=dataset_collection)


if __name__ == "__main__":
    yolov8_classification_training_pipeline()
