import os

from src import Pipeline, step
from src.models.contexts.picsellia_context import PicselliaTrainingContext


@step(name="Extract the weights", continue_on_failure=True)
def weights_extractor() -> str:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_file = context.experiment.get_artifact("weights")
    weights_destination_path = os.path.join(context.experiment.name, "weights")

    model_file.download(target_path=weights_destination_path)

    weights_path = os.path.join(weights_destination_path, model_file.filename)
    return os.path.abspath(weights_path)
