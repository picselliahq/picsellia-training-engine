from src import pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.yolov7.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from src.models.parameters.training.yolov7.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
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
from src.steps.model_evaluation.common.yolov7_model_evaluator import (
    yolov7_model_context_evaluator,
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
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ]
):
    return PicselliaTrainingContext(
        hyperparameters_cls=Yolov7HyperParameters,
        augmentation_parameters_cls=Yolov7AugmentationParameters,
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
    model_context = yolov7_model_context_trainer(
        model_context=model_context, dataset_collection=dataset_collection
    )

    yolov7_model_context_evaluator(
        model_context=model_context, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    yolov7_segmentation_training_pipeline()
