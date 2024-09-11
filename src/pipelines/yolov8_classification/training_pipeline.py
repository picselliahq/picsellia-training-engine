from src import pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.steps.data_extraction.training.training_data_extractor import (
    training_dataset_collection_extractor,
)
from src.steps.data_preparation.training.ultralytics_classification_data_preparator import (
    ultralytics_classification_dataset_collection_preparator,
)
from src.steps.data_validation.common.classification_data_validator import (
    classification_dataset_collection_validator,
)
from src.steps.model_evaluation.ultralytics_model_evaluator import (
    ultralytics_model_context_evaluator,
)
from src.steps.model_export.ultralytics_model_exporter import (
    ultralytics_model_context_exporter,
)
from src.steps.model_loading.common.ultralytics.ultralytics_model_context_loader import (
    ultralytics_model_context_loader,
)
from src.steps.model_training.ultralytics_trainer import (
    ultralytics_model_context_trainer,
)
from src.steps.weights_extraction.training.training_weights_extractor import (
    training_model_context_extractor,
)


def get_context() -> (
    PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ]
):
    return PicselliaTrainingContext(
        hyperparameters_cls=UltralyticsHyperParameters,
        augmentation_parameters_cls=UltralyticsAugmentationParameters,
        export_parameters_cls=ExportParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_classification_training_pipeline():
    dataset_collection = training_dataset_collection_extractor()
    dataset_collection = ultralytics_classification_dataset_collection_preparator(
        dataset_collection=dataset_collection
    )
    classification_dataset_collection_validator(dataset_collection=dataset_collection)

    model_context = training_model_context_extractor()
    model_context = ultralytics_model_context_loader(model_context=model_context)
    model_context = ultralytics_model_context_trainer(
        model_context=model_context, dataset_collection=dataset_collection
    )

    ultralytics_model_context_exporter(model_context=model_context)
    ultralytics_model_context_evaluator(
        model_context=model_context, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    yolov8_classification_training_pipeline()
