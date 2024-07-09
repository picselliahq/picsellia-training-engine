import os
from typing import List, Tuple, Union
from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from src.models.model.picsellia_prediction import (
    PicselliaRectangle,
    PicselliaOCRPrediction,
    PicselliaText,
    PicselliaConfidence,
    PicselliaLabel,
)
from src.models.steps.model_inferencing.model_collection_inference import (
    ModelCollectionInference,
)
from src.pipelines.paddle_ocr.PaddleOCR import PaddleOCR  # type: ignore[attr-defined]


def get_annotations_from_result(
    result,
) -> Tuple[List[PicselliaRectangle], List[PicselliaText], List[PicselliaConfidence]]:
    result = result[0]
    print(f"result: {result}")
    if not result:
        return [], [], []
    boxes = [get_picsellia_rectangle(line[0]) for line in result]
    texts = [get_picsellia_text(line[1][0]) for line in result]
    confidences = [get_picsellia_confidence(line[1][1]) for line in result]
    return boxes, texts, confidences


def get_picsellia_rectangle(points: List[List[int]]) -> PicselliaRectangle:
    x = min(point[0] for point in points)
    y = min(point[1] for point in points)
    w = max(point[0] for point in points) - x
    h = max(point[1] for point in points) - y
    return PicselliaRectangle(x, y, w, h)


def get_picsellia_text(text: str) -> PicselliaText:
    return PicselliaText(text)


def get_picsellia_confidence(confidence: float) -> PicselliaConfidence:
    return PicselliaConfidence(confidence)


class PaddleOCRModelCollectionInference(
    ModelCollectionInference[PaddleOCRModelCollection]
):
    def __init__(self, model_collection: PaddleOCRModelCollection):
        super().__init__(model_collection)
        self.model_collection: PaddleOCRModelCollection = model_collection
        self.model = self.load_model()

    def load_model(self) -> PaddleOCR:
        return PaddleOCR(
            use_angle_cls=True,
            rec_model_dir=self.model_collection.text_model.inference_model_path,
            det_model_dir=self.model_collection.bbox_model.inference_model_path,
            rec_char_dict_path=os.path.join(
                self.model_collection.text_model.model_weights_path, "en_dict.txt"
            ),
            use_gpu=True,
            show_log=False,
        )

    def get_evaluation(
        self, image_path: str, dataset_context: TDatasetContext
    ) -> Union[PicselliaOCRPrediction, None]:
        prediction = self.model.ocr(image_path)
        boxes, texts, confidences = get_annotations_from_result(prediction)
        if boxes:
            asset = dataset_context.dataset_version.find_all_assets(
                ids=[os.path.basename(image_path).split(".")[0]]
            )[0]
            classes = [
                PicselliaLabel(
                    dataset_context.dataset_version.get_or_create_label("text")
                )
                for _ in texts
            ]
            return PicselliaOCRPrediction(asset, boxes, classes, texts, confidences)
        return None

    def predict_on_dataset_context(
        self, dataset_context: TDatasetContext
    ) -> List[PicselliaOCRPrediction]:
        image_paths = [
            os.path.join(dataset_context.image_dir, image_name)
            for image_name in os.listdir(dataset_context.image_dir)
        ]
        evaluations = []
        for image_path in image_paths:
            evaluation = self.get_evaluation(image_path, dataset_context)
            if evaluation:
                evaluations.append(evaluation)
        return evaluations
