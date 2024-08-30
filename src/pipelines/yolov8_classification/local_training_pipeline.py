# type: ignore
from argparse import ArgumentParser

from src import pipeline
from src.models.contexts.training.test_picsellia_training_context import (
    TestPicselliaTrainingContext,
)
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
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
from src.steps.model_loading.common.ultralytics.ultralytics_model_context_loader import (
    ultralytics_model_context_loader,
)
from src.steps.model_training.ultralytics_trainer import ultralytics_trainer
from src.steps.weights_extraction.training.training_weights_extractor import (
    training_weights_extractor,
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
    dataset_collection = training_data_extractor()
    dataset_collection = ultralytics_classification_data_preparator(
        dataset_collection=dataset_collection
    )
    classification_data_validator(dataset_collection=dataset_collection)

    model_context = training_weights_extractor()
    model_context = ultralytics_model_context_loader(model_context=model_context)
    model_context = ultralytics_trainer(
        model_context=model_context, dataset_collection=dataset_collection
    )

    model_context = ultralytics_model_exporter(model_context=model_context)
    ultralytics_model_evaluator(
        model_context=model_context, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    import torch
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    yolov8_classification_training_pipeline()
