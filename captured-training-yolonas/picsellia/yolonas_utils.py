import os
from tqdm import tqdm
import shutil
import json
from picsellia import Client
from picsellia.types.enums import AnnotationFileType

import torch


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format:
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format:
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """

    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]


def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def convert_coco_json_to_yolo_txt(output_path, json_file):
    """
    Create folders containing dataset in yolo format 

    Args:
        output_path (str): 
        json_file (str): json file containing the coco annotations 

    Returns:
        output_path (str): dataset directory path
    """
    path = make_folders(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return output_path


def get_experiment():
    if 'api_token' not in os.environ:
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

    client = Client(
        api_token=api_token,
        host=host,
        organization_id=organization_id
    )

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
        raise Exception("You must set the project_token or project_name and experiment_name")
    return experiment


def convert_test_results(test_results):
    """convert test results to be logged

    Args:
        test_results (dict): test results 

    Returns:
        test_results: converted test results 
    """
    test_results['PPYoloELoss/loss_cls'] = float(test_results['PPYoloELoss/loss_cls'])
    test_results['PPYoloELoss/loss_iou'] = float(test_results['PPYoloELoss/loss_iou'])
    test_results['PPYoloELoss/loss_dfl'] = float(test_results['PPYoloELoss/loss_dfl'])
    test_results['PPYoloELoss/loss'] = float(test_results['PPYoloELoss/loss'])
    test_results['Precision@0.50'] = torch.Tensor.item(test_results['Precision@0.50'])
    test_results['Recall@0.50'] = torch.Tensor.item(test_results['Recall@0.50'])
    test_results['mAP@0.50'] = torch.Tensor.item(test_results['mAP@0.50'])
    test_results['F1@0.50'] = torch.Tensor.item(test_results['F1@0.50'])

    return test_results


def get_yolo_dataset(experiment, cwd):
    """download the dataset in coco format and convert it to yolo 
        save new data under cwd in new directory with the experiment's name 

    Args:
        experiment: current experiment
        cwd: current working directory

    Returns:
        path_dict (dict): dictionary containing relevant paths for the dataset 
        labelmap (list): current labelmap 
        label_names (dict): list of labels 
    """

    path_dict = {}
    for data_type in ['train', 'test', 'val']:
        path_dict[data_type] = {}
        path_dict[data_type]['dataset_dir'] = os.path.join(cwd, experiment.base_dir, data_type, 'images')

        assets = experiment.get_dataset(data_type)
        assets.download(path_dict[data_type]['dataset_dir'])

        # get the coco annotation from picsellia
        path_dict[data_type]['coco_annotation_path'] = assets.export_annotation_file(AnnotationFileType.COCO,
                                                                                    target_path=path_dict[data_type][
                                                                                        'dataset_dir'])

        # convert coco to yolo
        path_dict[data_type]['yolo_annotation_path'] = convert_coco_json_to_yolo_txt(
            output_path=os.path.join(experiment.base_dir, data_type, 'labels'),
            json_file=path_dict[data_type]['coco_annotation_path'])

        if data_type == 'train':
            labels = assets.list_labels()
            label_names = [label.name for label in labels]
            labelmap = {str(i): label.name for i, label in enumerate(labels)}

    print("Converting COCO to YOLO finished!")
    return path_dict, labelmap, label_names


def get_asset_predictions(experiment, model, asset, conf):
    """
        Get prediction's bouding boxes of one asset

    Args:
        experiment (_type_): 
        model: model to make inference with 
        asset: asset to make predictions on 
        conf (float): confidence score threshold below which bbox predictions are ignored 

    Returns:
        bbox_list (list): list containing the bounding boxes' info
        [List[Tuple[List[List[int]], Label, float]]]
    """

    labels = experiment.get_dataset("test").list_labels()
    image_path = os.path.join(experiment.base_dir, "test", "images", asset.filename)
    predictions = model.predict(image_path,  conf=0.1)
    bbox_list = []
    for image_prediction in predictions:
        class_names = image_prediction.class_names
        class_ids = image_prediction.prediction.labels
        confidence = image_prediction.prediction.confidence
        bboxes = image_prediction.prediction.bboxes_xyxy

        for i, (label, conf, bbox) in enumerate(zip(class_ids, confidence, bboxes)):
            for i, coord in enumerate(bbox):
                if coord < 0:
                    bbox[i] = 0
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_list.append((int(bbox[0]), int(bbox[1]), int(width), int(height), labels[int(label)], float(conf)))
    
    return bbox_list 