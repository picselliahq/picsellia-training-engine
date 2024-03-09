from ultralytics.models.yolo.classify import ClassificationTrainer

from poc.step import step


@step
def model_loader(
    context: dict, checkpoints_path: str, dataset_path: str, augmentation_args: dict
):
    model_args = dict(
        model=checkpoints_path,
        project=context["experiment"].name,
        data=dataset_path,
        **context["parameters"],
        **augmentation_args,
    )
    trainer = ClassificationTrainer(overrides=model_args)
    return trainer
