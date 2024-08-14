from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.steps.model_export.ultralytics_model_context_exporter import (
    UltralyticsModelContextExporter,
)


@step
def ultralytics_model_exporter(model_context):
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    model_context_exporter = UltralyticsModelContextExporter(
        model_context=model_context, experiment=context.experiment
    )
    model_context = model_context_exporter.export_and_save_model_context()
    return model_context
