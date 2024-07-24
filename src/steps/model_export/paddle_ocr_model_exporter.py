from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.steps.model_export.paddle_ocr_model_collection_exporter import (
    PaddleOCRModelCollectionExporter,
)


@step
def paddle_ocr_model_exporter(model_collection):
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    model_collection_exporter = PaddleOCRModelCollectionExporter(
        model_collection=model_collection, experiment=context.experiment
    )
    model_collection = model_collection_exporter.export_and_save_model_collection()
    return model_collection
