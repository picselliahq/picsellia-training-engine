from ultralytics import YOLO

from src.models.contexts.picsellia_context import PicselliaTrainingContext
from src import Pipeline
from src import step


@step
def model_exporter(model: YOLO):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    exported_model = model.export(**context["exporter_args"])
    return exported_model
