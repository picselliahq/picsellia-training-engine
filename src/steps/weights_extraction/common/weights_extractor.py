from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.model_context import ModelContext
from src.models.steps.weights_extraction.training.training_model_context_extractor import (
    TrainingModelContextExtractor,
)


@step
def weights_extractor() -> ModelContext:
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    model_context_extractor = TrainingModelContextExtractor(
        experiment=context.experiment
    )
    model_context = model_context_extractor.get_model_context()
    model_context.download_weights()
    return model_context
