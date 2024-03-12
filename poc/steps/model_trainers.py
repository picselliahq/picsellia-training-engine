from ultralytics import YOLO

from poc.step import step


@step
def model_trainer(context: dict, model: YOLO, callbacks: dict, dataset_path: str):
    trainer_args = dict(
        project=context["experiment"].name,
        data=dataset_path,
        **context["training_args"],
        **context["augmentation_args"],
    )
    for callback_name, callback_function in callbacks.items():
        model.add_callback(event=callback_name, func=callback_function)
    model.train(**trainer_args)
    return model
