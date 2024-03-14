from poc.models.contexts.picsellia_context import PicselliaTrainingContext
from poc.pipeline import Pipeline
from poc.step import step


@step
def model_register(weights_name: str, weights_path: str):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    artifact = context.experiment.store(weights_name, weights_path)
    return artifact
