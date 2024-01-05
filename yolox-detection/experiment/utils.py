import logging
import math
import os
from typing import List, Union

import numpy as np
import picsellia
import requests
import torch
import tqdm
from PIL import Image
from PIL import UnidentifiedImageError
from picsellia import Job, DatasetVersion, Asset
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import InferenceType
from pycocotools.coco import COCO

from YOLOX.tools.demo import Predictor
from evaluator.type_formatter import TypeFormatter
from evaluator.utils.general import transpose_if_exif_tags


class YOLOV8StyleOutput:
    class Boxes:
        def __init__(self, boxes, conf, cls):
            self.xyxyn = boxes
            self.conf = conf
            self.cls = cls

    def __init__(self, yolox_output: list, img_info: dict):
        self.img_width = img_info.get("width", 1)
        self.img_height = img_info.get("height", 1)
        self.ratio = img_info.get("ratio", 1)

        # Extract and normalize boxes
        boxes = yolox_output[:, 0:4]
        normalized_boxes = torch.zeros_like(boxes)
        normalized_boxes[:, 0] = boxes[:, 0] / self.img_width
        normalized_boxes[:, 1] = boxes[:, 1] / self.img_height
        normalized_boxes[:, 2] = boxes[:, 2] / self.img_width
        normalized_boxes[:, 3] = boxes[:, 3] / self.img_height

        # Extract confidences and classes
        conf = yolox_output[:, 4] * yolox_output[:, 5]
        cls = yolox_output[:, 6]

        # Create Boxes object
        self.boxes = self.Boxes(normalized_boxes, conf, cls)

    @property
    def probs(self):
        return self.boxes.conf


def get_experiment() -> Experiment:
    if "api_token" not in os.environ:
        raise Exception("You must set an api_token to run this image")
    api_token = os.environ["api_token"]

    if "host" not in os.environ:
        host = "https://app.picsellia.com"
    else:
        host = os.environ["host"]

    if "organization_id" not in os.environ:
        organization_id = None
    else:
        organization_id = os.environ["organization_id"]

    client = picsellia.Client(
        api_token=api_token, host=host, organization_id=organization_id
    )

    if "experiment_id" in os.environ:
        experiment_id = os.environ["experiment_id"]

        experiment = client.get_experiment_by_id(experiment_id)
    else:
        raise Exception("You must set the experiment_id")
    return experiment


def create_yolo_detection_label(
    experiment: Experiment,
    data_type: str,
    annotations_dict: dict,
    annotations_coco: COCO,
    label_names: list,
):
    dataset_path = os.path.join(experiment.png_dir, data_type)
    image_filenames = os.listdir(os.path.join(dataset_path, "images"))

    labels_path = os.path.join(dataset_path, "labels")

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for img in annotations_dict["images"]:
        img_filename = img["file_name"]
        if img_filename in image_filenames:
            create_img_label_detection(img, annotations_coco, labels_path, label_names)


def create_img_label_detection(
    image: dict, annotations_coco: COCO, labels_path: str, label_names: list
):
    result = []
    img_id = image["id"]
    img_filename = image["file_name"]
    w = image["width"]
    h = image["height"]
    txt_name = os.path.splitext(img_filename)[0] + ".txt"
    annotation_ids = annotations_coco.getAnnIds(imgIds=img_id)
    anns = annotations_coco.loadAnns(annotation_ids)
    for ann in anns:
        bbox = ann["bbox"]
        yolo_bbox = coco_to_yolo_detection(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        seg_string = " ".join([str(x) for x in yolo_bbox])
        label = label_names.index(
            annotations_coco.loadCats(ann["category_id"])[0]["name"]
        )
        result.append(f"{label} {seg_string}")
    with open(os.path.join(labels_path, txt_name), "w") as f:
        f.write("\n".join(result))


def coco_to_yolo_detection(
    x1: int, y1: int, w: int, h: int, image_w: int, image_h: int
) -> list[float]:
    return [
        ((2 * x1 + w) / (2 * image_w)),
        ((2 * y1 + h) / (2 * image_h)),
        w / image_w,
        h / image_h,
    ]


def evaluate_model(
    yolox_predictor: Predictor,
    type_formatter: TypeFormatter,
    experiment: Experiment,
    dataset: DatasetVersion,
    confidence_threshold: float = 0.1,
) -> Job:
    asset_list = dataset.list_assets()
    evaluation_batch_size = experiment.get_log(name="parameters").data.get(
        "evaluation_batch_size", 8
    )
    batch_size = (
        evaluation_batch_size
        if len(asset_list) > evaluation_batch_size
        else len(asset_list)
    )
    total_batch_number = math.ceil(len(asset_list) / batch_size)

    for i in tqdm.tqdm(range(total_batch_number)):
        subest_asset_list = asset_list[i * batch_size : (i + 1) * batch_size]
        inputs = preprocess_images(subest_asset_list)

        for j, asset in enumerate(subest_asset_list):
            prediction, img_info = yolox_predictor.inference(inputs[j])

            if prediction[0] is not None:
                yolov8_style_output = YOLOV8StyleOutput(
                    yolox_output=prediction[0], img_info=img_info
                )

                evaluations = format_prediction_to_evaluations(
                    asset=asset,
                    prediction=yolov8_style_output,
                    type_formatter=type_formatter,
                    confidence_threshold=confidence_threshold,
                )
                send_evaluations_to_platform(
                    experiment=experiment, asset=asset, evaluations=evaluations
                )

    if dataset.type in [
        InferenceType.OBJECT_DETECTION,
        InferenceType.SEGMENTATION,
        InferenceType.CLASSIFICATION,
    ]:
        return experiment.compute_evaluations_metrics(inference_type=dataset.type)


def preprocess_images(assets: List[Asset]) -> List[np.array]:
    images = []
    for asset in assets:
        try:
            image = open_asset_as_array(asset)
        except UnidentifiedImageError:
            logging.warning(f"Can't evaluate {asset.filename}, error opening the image")
            continue
        images.append(image)
    return images


def open_asset_as_array(asset: Asset) -> np.array:
    image = Image.open(requests.get(asset.reset_url(), stream=True).raw)
    image = transpose_if_exif_tags(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def send_evaluations_to_platform(
    experiment: Experiment, asset: Asset, evaluations: List
) -> None:
    shapes = {"rectangles": evaluations}
    try:
        experiment.add_evaluation(asset=asset, **shapes)
        print(f"Asset: {asset.filename} evaluated.")
        logging.info(f"Asset: {asset.filename} evaluated.")
    except Exception:
        logging.info(
            f"Something went wrong with evaluating {asset.filename}. Skipping.."
        )


def format_prediction_to_evaluations(
    asset: Asset,
    prediction: Union[List, dict],
    type_formatter: TypeFormatter,
    confidence_threshold: float,
) -> List:
    picsellia_predictions = type_formatter.format_prediction(
        asset=asset, prediction=prediction
    )

    evaluations = []
    for i in range(min(100, len(picsellia_predictions["confidences"]))):
        if picsellia_predictions["confidences"][i] >= confidence_threshold:
            picsellia_prediction = {
                prediction_key: prediction[i]
                for prediction_key, prediction in picsellia_predictions.items()
            }
            evaluation = type_formatter.format_evaluation(
                picsellia_prediction=picsellia_prediction
            )
            evaluations.append(evaluation)
    return evaluations
