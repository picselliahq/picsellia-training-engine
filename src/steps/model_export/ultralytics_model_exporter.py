from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_export.ultralytics_model_context_exporter import (
    UltralyticsModelContextExporter,
)


@step
def ultralytics_model_exporter(model_context: ModelContext):
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    model_context_exporter = UltralyticsModelContextExporter(
        model_context=model_context, experiment=context.experiment
    )
    model_context_exporter.export_model_context(
        exported_model_destination_path=model_context.inference_model_dir,
        export_format=context.export_parameters.export_format,
    )
    model_context_exporter.save_model_to_experiment(
        exported_model_dir=model_context.inference_model_dir,
        saved_model_name="model-latest",
    )
