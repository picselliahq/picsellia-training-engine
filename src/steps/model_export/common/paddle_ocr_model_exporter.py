# type: ignore

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.paddle_ocr.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.paddle_ocr.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from src.models.parameters.training.paddle_ocr.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from src.models.steps.model_export.training.paddle_ocr_model_collection_exporter import (
    PaddleOCRModelCollectionExporter,
)


@step
def paddle_ocr_model_collection_exporter(
    model_collection: PaddleOCRModelCollection,
) -> PaddleOCRModelCollection:
    """
    Exports a PaddleOCR model collection and saves it to an experiment.

    This function retrieves the active training context from the pipeline, exports the provided
    PaddleOCR model collection in the specified format, and saves the exported models to the experiment.
    The `PaddleOCRModelCollectionExporter` is used to handle the export and save operations.

    Args:
        model_collection (PaddleOCRModelCollection): The PaddleOCR model collection to be exported.

    Returns:
        PaddleOCRModelCollection: The exported PaddleOCR model collection.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()
    model_collection_exporter = PaddleOCRModelCollectionExporter(
        model_collection=model_collection, experiment=context.experiment
    )
    model_collection = model_collection_exporter.export_model_collection(
        export_format=context.export_parameters.export_format
    )
    model_collection_exporter.save_model_collection()

    return model_collection
