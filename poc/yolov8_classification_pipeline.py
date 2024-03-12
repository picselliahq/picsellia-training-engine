from poc.pipeline import pipeline
from poc.steps.augmentation_preparators import augmentation_preparator
from poc.steps.checkpoint_preparators import checkpoints_preparator
from poc.steps.checkpoints_extractors import checkpoints_extractor
from poc.steps.checkpoints_validators import checkpoints_validator
from poc.steps.context_preparators import context_preparator
from poc.steps.data_extractors import data_extractor
from poc.steps.data_preparators import data_preparator
from poc.steps.model_loaders import model_loader
from poc.steps.model_trainers import model_trainer


@pipeline(log_folder_path="logs", remove_logs_on_completion=False)
def yolov8_classification_pipeline():
    api_token = "b2a2ffd0f4be0c79d0a719bf0d1177b5a12854eb"
    host = "https://app.picsellia.com"
    organization_name = "SoniaGrh"
    experiment_id = "018e198c-45ef-7a02-b7d8-3e152657de53"

    context = context_preparator(
        api_token=api_token,
        host=host,
        organization_name=organization_name,
        experiment_id=experiment_id,
    )

    # Data pipeline
    dataset_context = data_extractor(context=context)
    augmentation_args = augmentation_preparator(context)
    data_path = data_preparator(context=context, dataset_context=dataset_context)

    # Model pipeline
    checkpoints_path = checkpoints_extractor(context)
    checkpoints_path = checkpoints_preparator(context, checkpoints_path)
    checkpoints_path = checkpoints_validator(context, checkpoints_path)

    trainer = model_loader(context, checkpoints_path, data_path, augmentation_args)
    model_trainer(trainer)


if __name__ == "__main__":
    yolov8_classification_pipeline()
