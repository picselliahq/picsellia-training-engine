from ultralytics import YOLO

from src import Pipeline, step
from src.models.contexts.picsellia_context import PicselliaTrainingContext


@step
def model_exporter(model: YOLO):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    exported_model = model.export(**context["exporter_args"])
    return exported_model
