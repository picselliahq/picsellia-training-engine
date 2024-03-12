from ultralytics import YOLO

from poc.step import step


@step
def model_exporter(context: dict, model: YOLO):
    exported_model = model.export(**context["exporter_args"])
    return exported_model
