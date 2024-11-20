import logging

from src import Pipeline, step
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_export.training.yolox_model_context_exporter import (
    YoloXModelContextExporter,
)

logger = logging.getLogger(__name__)


@step
def yolox_model_context_exporter(model_context: ModelContext):
    """
    Exports and saves the YoloX model context to the experiment.

    This function retrieves the active training context from the pipeline, exports the YoloX model
    in the specified format, and saves the exported model weights to the experiment. If the `exported_weights_dir`
    is not found in the model context, the export process is skipped, and a log message is generated.

    Args:
        model_context (ModelContext): The model context for the Ultralytics model to be exported and saved.

    Raises:
        If no `exported_weights_dir` is found in the model context, the export process is skipped, and
        a log message is generated.
    """
    context = Pipeline.get_active_context()

    model_context_exporter = YoloXModelContextExporter(model_context=model_context)

    if model_context.exported_weights_dir:
        export_format = context.export_parameters.export_format.value
        export_filename = context.export_parameters.exported_file_name

        exported_model_path = model_context_exporter.export_model_context(
            exported_weights_destination_path=model_context.exported_weights_dir,
            export_format=export_format,
        )
        model_context_exporter.save_model_to_model_version(
            model_version=model_context.model_version,
            exported_weights_path=exported_model_path,
            exported_weights_name=export_filename,
        )
    else:
        logger.info(
            "No exported weights directory found in model context. Skipping export."
        )
