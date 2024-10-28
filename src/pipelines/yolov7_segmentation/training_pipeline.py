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
from src.steps.data_preparation.training.yolov7_data_preparator import (
    yolov7_dataset_collection_preparator,
)
from src.steps.data_validation.common.segmentation_dataset_collection_validator import (
    segmentation_dataset_collection_validator,
)
from src.steps.model_training.yolov7_trainer import yolov7_model_context_trainer
from src.steps.weights_extraction.training.yolov7_weights_extractor import (
    yolov7_model_context_extractor,
)
from src.steps.weights_preparation.training.yolov7_weights_preparator import (
    yolov7_model_context_preparator,
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
def yolov7_segmentation_training_pipeline():
    dataset_collection = training_dataset_collection_extractor()
    dataset_collection = yolov7_dataset_collection_preparator(
        dataset_collection=dataset_collection
    )
    segmentation_dataset_collection_validator(
        dataset_collection=dataset_collection, fix_annotation=True
    )

    model_context = yolov7_model_context_extractor()
    model_context = yolov7_model_context_preparator(model_context=model_context)
    yolov7_model_context_trainer(
        model_context=model_context, dataset_collection=dataset_collection
    )
    #
    # ultralytics_model_context_exporter(model_context=model_context)
    # ultralytics_model_context_evaluator(
    #     model_context=model_context, dataset_context=dataset_collection["test"]
    # )


if __name__ == "__main__":
    yolov7_segmentation_training_pipeline()
