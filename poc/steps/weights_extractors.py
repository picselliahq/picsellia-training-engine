import os


from poc.models.contexts.picsellia_context import PicselliaTrainingContext
from poc.pipeline import Pipeline
from poc.step import step


@step(name="Extract the weights", continue_on_failure=True)
def weights_extractor() -> str:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_file = context.experiment.get_artifact("weights")
    destination_path = os.path.join(context.experiment.name, "weights")

    model_file.download(target_path=destination_path)

    return os.path.abspath(
        os.path.join(destination_path, "weights", model_file.filename)
    )
