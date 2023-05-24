import os
from tqdm import tqdm
import shutil
import json
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from picsellia.sdk.experiment import Experiment, Asset
import torch


def get_experiment():
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

    client = Client(api_token=api_token, host=host, organization_id=organization_id)

    if "experiment_name" in os.environ:
        experiment_name = os.environ["experiment_name"]
        if "project_token" in os.environ:
            project_token = os.environ["project_token"]
            project = client.get_project_by_id(project_token)
        elif "project_name" in os.environ:
            project_name = os.environ["project_name"]
            project = client.get_project(project_name)
        experiment = project.get_experiment(experiment_name)
    else:
        raise Exception(
            "You must set the project_token or project_name and experiment_name"
        )
    return experiment


def get_labelmap(experiment: Experiment):
    labels = experiment.get_dataset("train").list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i): label.name for i, label in enumerate(labels)}
    return label_names, labelmap


def format_test_results(test_results):
    test_results["PPYoloELoss/loss_cls"] = float(test_results["PPYoloELoss/loss_cls"])
    test_results["PPYoloELoss/loss_iou"] = float(test_results["PPYoloELoss/loss_iou"])
    test_results["PPYoloELoss/loss_dfl"] = float(test_results["PPYoloELoss/loss_dfl"])
    test_results["PPYoloELoss/loss"] = float(test_results["PPYoloELoss/loss"])
    test_results["Precision@0.50"] = torch.Tensor.item(test_results["Precision@0.50"])
    test_results["Recall@0.50"] = torch.Tensor.item(test_results["Recall@0.50"])
    test_results["mAP@0.50"] = torch.Tensor.item(test_results["mAP@0.50"])
    test_results["F1@0.50"] = torch.Tensor.item(test_results["F1@0.50"])

    return test_results


def create_yolo_dataset(experiment: Experiment, cwd):
    path_dict = {}
    for data_type in ["train", "test", "val"]:
        path_dict[data_type] = {}
        path_dict[data_type]["dataset_dir"] = os.path.join(
            cwd, experiment.base_dir, data_type, "images"
        )

        assets = experiment.get_dataset(data_type)
        assets.download(path_dict[data_type]["dataset_dir"])

        # get the coco annotation from picsellia
        path_dict[data_type]["coco_annotation_path"] = assets.export_annotation_file(
            AnnotationFileType.COCO, target_path=path_dict[data_type]["dataset_dir"]
        )

        # convert coco to yolo
        path_dict[data_type]["yolo_annotation_path"] = convert_coco_json_to_yolo_txt(
            output_path=os.path.join(experiment.base_dir, data_type, "labels"),
            json_file=path_dict[data_type]["coco_annotation_path"],
        )

    return path_dict


def convert_coco_json_to_yolo_txt(output_path: str, json_file: str):
    path = make_folders(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
        annotations = [
            annotation
            for annotation in json_data["annotations"]
            if annotation["image_id"] == image["id"]
        ]
        annotation_txt = os.path.join(
            output_path, image["file_name"].split(".")[0] + ".txt"
        )
        with open(annotation_txt, "w") as f:
            for annotation in annotations:
                category = annotation["category_id"]
                x, y, w, h = convert_bbox_coco2yolo(
                    image["width"], image["height"], annotation["bbox"]
                )
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return output_path


def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center / img_width
    y = y_center / img_height
    w = w / img_width
    h = h / img_height

    return [x, y, w, h]


def get_asset_predictions(
    experiment: Experiment, model, asset, conf_threshold, dataset_type: str
):
    image_path = os.path.join(
        experiment.base_dir, dataset_type, "images", asset.filename
    )
    return model.predict(image_path, conf=conf_threshold)


def format_asset_predictions_for_eval(predictions, experiment_labels):
    bbox_list = []
    for image_prediction in predictions:
        for i, (label, conf, bbox) in enumerate(
            zip(
                image_prediction.prediction.labels,
                image_prediction.prediction.confidence,
                image_prediction.prediction.bboxes_xyxy,
            )
        ):
            for i, coord in enumerate(bbox):
                if coord < 0:
                    bbox[i] = 0
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_list.append(
                (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(width),
                    int(height),
                    experiment_labels[int(label)],
                    float(conf),
                )
            )

    return bbox_list
