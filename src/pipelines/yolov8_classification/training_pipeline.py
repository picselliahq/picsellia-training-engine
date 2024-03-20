from src import pipeline
from src.models.contexts.picsellia_context import (
    PicselliaTrainingContext,
    PicselliaContext,
)
from src.models.parameters.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.hyper_parameters import UltralyticsHyperParameters
from src.steps.callbacks_preparators import callback_preparator
from src.steps.data_validators import data_validator
from src.steps.weights_preparators import weights_preparator
from src.steps.weights_extractors import weights_extractor
from src.steps.weights_validators import weights_validator
from src.steps.data_extractors import data_extractor
from src.steps.data_preparators import data_preparator
from src.steps.model_evaluators import model_evaluator
from src.steps.model_exporters import model_exporter
from src.steps.model_inferences import model_inference
from src.steps.model_loaders import model_loader
from src.steps.model_registers import model_register
from src.steps.model_trainers import model_trainer


def get_context() -> PicselliaContext:
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
    dataset_context = data_extractor()
    dataset_context = data_validator(dataset_context=dataset_context)
    data_path = data_preparator(dataset_context=dataset_context)

    # Model pipeline
    weights_path = weights_extractor()
    weights_path = weights_validator(weights_path=weights_path)
    weights_path = weights_preparator(weights_path=weights_path)
    model = model_loader(weights_path=weights_path)

    # Training pipeline
    callbacks = callback_preparator()
    model = model_trainer(model=model, callbacks=callbacks, dataset_path=data_path)
    exported_model_path = model_exporter(model=model)
    _ = model_register(
        weights_name="model-latest-onnx",
        weights_path=str(exported_model_path),
    )

    # Evaluation pipeline
    picsellia_predictions = model_inference(
        model=model,
        dataset_context=dataset_context,
        attached_dataset_version="test",
    )
    _ = model_evaluator(
        dataset_context=dataset_context,
        attached_dataset_version="test",
        picsellia_predictions=picsellia_predictions,
    )


if __name__ == "__main__":
    yolov8_classification_training_pipeline()
