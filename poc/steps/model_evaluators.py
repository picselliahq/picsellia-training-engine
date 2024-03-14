import os

from poc.models.contexts.picsellia_context import PicselliaTrainingContext
from poc.pipeline import Pipeline
from poc.step import step


@step
def model_evaluator(
    dataset_context: dict,
    attached_dataset_version: str,
    picsellia_predictions: list[tuple],
):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    for image_path, image_predictions in zip(
        dataset_context[attached_dataset_version]["images_list"], picsellia_predictions
    ):
        asset = context.experiment.get_dataset(
            attached_dataset_version
        ).find_all_assets(ids=[os.path.basename(image_path).split(".")[0]])[0]

        context.experiment.add_evaluation(
            asset=asset, classifications=[image_predictions]
        )

    return True
