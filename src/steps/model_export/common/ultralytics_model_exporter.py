from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_export.training.ultralytics_model_context_exporter import (
    UltralyticsModelContextExporter,
)
import logging

logger = logging.getLogger(__name__)


@step
def ultralytics_model_context_exporter(model_context: ModelContext):
    """
    Exports and saves the Ultralytics model context to the experiment.

    This function retrieves the active training context from the pipeline, exports the Ultralytics model
    in the specified format, and saves the exported model weights to the experiment. If the `exported_weights_dir`
    is not found in the model context, the export process is skipped, and a log message is generated.

    Args:
        model_context (ModelContext): The model context for the Ultralytics model to be exported and saved.

    Raises:
        If no `exported_weights_dir` is found in the model context, the export process is skipped, and
        a log message is generated.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_exporter = UltralyticsModelContextExporter(
        model_context=model_context, experiment=context.experiment
    )

    if model_context.exported_weights_dir:
        model_context_exporter.export_model_context(
            exported_weights_destination_path=model_context.exported_weights_dir,
            export_format=context.export_parameters.export_format,
        )
        model_context_exporter.save_model_to_experiment(
            exported_weights_dir=model_context.exported_weights_dir,
            exported_weights_name="model-latest",
        )
    else:
        logger.info(
            "No exported weights directory found in model context. Skipping export."
        )
