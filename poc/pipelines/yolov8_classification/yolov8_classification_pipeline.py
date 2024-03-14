import os

from poc.pipeline import pipeline
from poc.steps.callbacks_preparators import callback_preparator
from poc.steps.data_validators import data_validator
from poc.steps.weights_preparators import weights_preparator
from poc.steps.weights_extractors import weights_extractor
from poc.steps.weights_validators import weights_validator
from poc.steps.context_preparators import context_preparator
from poc.steps.data_extractors import data_extractor
from poc.steps.data_preparators import data_preparator
from poc.steps.model_evaluators import model_evaluator
from poc.steps.model_exporters import model_exporter
from poc.steps.model_inferences import model_inference
from poc.steps.model_loaders import model_loader
from poc.steps.model_registers import model_register
from poc.steps.model_trainers import model_trainer

api_token = os.environ["api_token"]
host = "https://app.picsellia.com"
organization_name = "SoniaGrh"
experiment_id = "018e3238-b56e-7f2a-b118-222a039ce80a"


@pipeline(log_folder_path="logs/", remove_logs_on_completion=False)
def yolov8_classification_pipeline():
    context = context_preparator(
        api_token=api_token,
        host=host,
        organization_name=organization_name,
        experiment_id=experiment_id,
    )

    # Data pipeline
    dataset_context = data_extractor(context=context)
    dataset_context = data_validator(context=context, dataset_context=dataset_context)
    data_path = data_preparator(context=context, dataset_context=dataset_context)

    # Model pipeline
    weights_path = weights_extractor(context=context)
    weights_path = weights_validator(context=context, weights_path=weights_path)
    weights_path = weights_preparator(context=context, weights_path=weights_path)
    model = model_loader(context=context, weights_path=weights_path)

    # Training pipeline
    callbacks = callback_preparator(context=context)
    model = model_trainer(
        context=context, model=model, callbacks=callbacks, dataset_path=data_path
    )
    exported_model_path = model_exporter(context=context, model=model)
    _ = model_register(
        context=context,
        weights_name="model-latest-onnx",
        weights_path=str(exported_model_path),
    )

    # Evaluation pipeline
    picsellia_predictions = model_inference(
        context=context,
        model=model,
        dataset_context=dataset_context,
        attached_dataset_version="test",
    )
    _ = model_evaluator(
        context=context,
        dataset_context=dataset_context,
        attached_dataset_version="test",
        picsellia_predictions=picsellia_predictions,
    )


if __name__ == "__main__":
    yolov8_classification_pipeline()
