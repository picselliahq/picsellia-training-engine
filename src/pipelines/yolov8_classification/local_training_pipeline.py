# type: ignore
from argparse import ArgumentParser

from src import pipeline
from src.models.contexts.training.test_picsellia_training_context import (
    TestPicselliaTrainingContext,
)
from src.models.parameters.common.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.common.hyper_parameters import UltralyticsHyperParameters
from src.steps.data_extraction.training.training_data_extractor import (
    training_data_extractor,
)
from src.steps.data_preparation.common.classification_data_preparator import (
    classification_data_preparator,
)
from src.steps.data_validation.common.classification_data_validator import (
    classification_data_validator,
)

parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--experiment_id", type=str)

args = parser.parse_args()


def get_context() -> TestPicselliaTrainingContext:
    return TestPicselliaTrainingContext(
        api_token=args.api_token,
        organization_id=args.organization_id,
        experiment_id=args.experiment_id,
        hyperparameters_cls=UltralyticsHyperParameters,
        augmentation_parameters_cls=UltralyticsAugmentationParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_classification_training_pipeline():
    dataset_collection = training_data_extractor(random_seed=42)
    dataset_collection = classification_data_preparator(
        dataset_collection=dataset_collection
    )
    classification_data_validator(dataset_collection=dataset_collection)


if __name__ == "__main__":
    yolov8_classification_training_pipeline()
