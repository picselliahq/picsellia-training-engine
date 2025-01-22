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
from src.steps.data_extraction.training.coco_data_extractor import (
    get_coco_dataset_collection,
)
from src.steps.data_preparation.training.ultralytics_classification_data_preparator import (
    prepare_ultralytics_classification_dataset_collection,
)
from src.steps.data_validation.common.coco_classification_dataset_collection_validator import (
    validate_coco_classification_dataset_collection,
)
from src.steps.model_evaluation.common.ultralytics_model_evaluator import (
    evaluate_ultralytics_model_context,
)
from src.steps.model_export.common.ultralytics_model_exporter import (
    export_ultralytics_model_context,
)
from src.steps.model_loading.common.ultralytics.ultralytics_model_context_loader import (
    load_ultralytics_model_context,
)
from src.steps.model_training.ultralytics_trainer import train_ultralytics_model_context

from src.steps.weights_extraction.training.ultralytics_weights_extractor import (
    get_ultralytics_model_context,
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
    dataset_collection = get_coco_dataset_collection()
    prepare_ultralytics_classification_dataset_collection(
        dataset_collection=dataset_collection
    )
    validate_coco_classification_dataset_collection(
        dataset_collection=dataset_collection
    )

    model_context = get_ultralytics_model_context(
        pretrained_weights_name="pretrained-weights"
    )
    load_ultralytics_model_context(
        model_context=model_context,
        weights_path_to_load=model_context.pretrained_weights_path,
    )
    train_ultralytics_model_context(
        model_context=model_context, dataset_collection=dataset_collection
    )

    export_ultralytics_model_context(model_context=model_context)
    evaluate_ultralytics_model_context(
        model_context=model_context, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    yolov8_classification_training_pipeline()
