from picsellia import Label
from ultralytics import YOLO

from poc.models.contexts.picsellia_context import PicselliaTrainingContext
from poc.pipeline import Pipeline
from poc.step import step


def postprocess(results: list, labelmap: dict[str, Label]) -> list[tuple[Label, float]]:
    classifications = []
    for result in results:
        label_names = result.names
        label_id = result.probs.top1
        label_name = label_names[label_id]
        confidence = float(result.probs.top1conf.item())
        picsellia_label = labelmap[label_name]
        classifications.append((picsellia_label, confidence))
    return classifications


@step
def model_inference(
    model: YOLO, dataset_context: dict, attached_dataset_version: str
) -> list[tuple[Label, float]]:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    images_list = dataset_context[attached_dataset_version]["images_list"]
    labelmap = dataset_context[attached_dataset_version]["labelmap"]

    results = model.predict(images_list, **context["inference_args"])

    picsellia_results = postprocess(results, labelmap)
    return picsellia_results
