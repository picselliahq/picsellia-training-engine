import torchvision
from PIL import Image
import picsellia
from pycocotools.coco import COCO
import json
from datasets import DatasetDict
from picsellia.types.enums import AnnotationFileType, InferenceType
import numpy as np
import os
import albumentations
from tqdm import tqdm
import torch
import evaluate
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia import Experiment


def download_data(experiment: Experiment) -> tuple[DatasetVersion, str]:
    dataset_list = experiment.list_attached_dataset_versions()
    if len(dataset_list) == 1:
        dataset = dataset_list[0]
        data_dir = os.path.join(experiment.base_dir, "data")
        dataset.download(data_dir)

    return dataset, data_dir


def get_experiment() -> Experiment:
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

    client = picsellia.Client(
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
        Exception(
            "You must set the project_token or project_name and experiment_name")
    return experiment


transform = albumentations.Compose(
    [
        albumentations.Resize(960, 960),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),

    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


def get_category_mapping(annotations: dict) -> list[str]:
    return [cat['name'] for cat in annotations['categories']]


def read_annotation_file(dataset, target_path) -> tuple[dict, str]:
    annotation_file_path = dataset.export_annotation_file(annotation_file_type=AnnotationFileType.COCO,
                                                          target_path=target_path)
    with open(annotation_file_path) as f:
        annotations = json.load(f)
    return annotations, annotation_file_path


def create_objects_dict(annotations, image_id) -> dict:
    curr_object = {key: [] for key in ['id', 'bbox', 'category', 'area', 'image_id']}

    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            curr_object['id'].append(ann['id'])
            curr_object['bbox'].append(ann['bbox'])
            curr_object['category'].append(ann['category_id'])
            curr_object['area'].append(ann['area'])
            curr_object['image_id'].append(image_id)

    return curr_object


def format_coco_annot_to_jsonlines_format(annotations) -> list[dict]:
    formatted_coco = []
    for image in annotations['images']:
        image_id = image['id']
        one_line = {'file_name': image['file_name'], 'image_id': image_id, 'width': image['width'],
                    'height': image['height'], 'objects': create_objects_dict(annotations, image_id)}
        formatted_coco.append(one_line)

    return formatted_coco


def write_metadata_file(data, output_path):
    with open(output_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')


def custom_train_test_eval_split(loaded_dataset: DatasetDict, test_prop: float) -> DatasetDict:
    first_split = loaded_dataset['train'].train_test_split(test_size=test_prop)
    test_valid = first_split['test'].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict({
        'train': first_split['train'],
        'test': test_valid['test'],
        'eval': test_valid['train']
    })
    return train_test_valid_dataset


def formatted_anns(image_id, category, area, bbox):
    annotations = []

    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


# transforming a batch
def transform_aug_ann(examples, image_processor):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch, image_processor):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": labels}
    return batch


def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}


# Save images and annotations into the files torchvision.datasets.CocoDetection expects
def save_annotation_file_images(dataset, experiment, id2label):
    output_json = {}
    path_output = os.path.join(experiment.base_dir, "output")

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    path_anno = os.path.join(path_output, "ann.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    for example in dataset:
        ann = val_formatted_anns(example["image_id"], example["objects"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height": example["image"].height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    for im, img_id in zip(dataset["image"], dataset["image_id"]):
        path_img = os.path.join(path_output, f"{img_id}.png")
        im.save(path_img)

    return path_output, path_anno


def format_evaluation_results(results: dict) -> dict:
    casted_results = {}
    for metric, value in results['iou_bbox'].items():
        casted_results[metric] = float(value)

    return casted_results


def run_evaluation(test_ds_coco_format, im_processor, model):
    test_ds_coco_format = CocoDetection(path_output, image_processor, path_anno)

    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]

            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized

            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api

            module.add(prediction=results, reference=labels)
            del batch

    results = module.compute()
    return results


def predict_image(image_path: str, threshold: float, image_processor, model):
    with torch.no_grad():
        image = Image.open(image_path)
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = \
            image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
        # box format in results is: top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    return results


def get_dataset_image_ids(dataset, dataset_type: str) -> list:
    # dataset_type is either train, test or eval
    image_id_list = []
    for example in dataset[dataset_type]:
        image_id_list.append(example['image_id'])
    return image_id_list


def get_filenames_by_ids(image_ids: list, annotations: dict, id_list) -> dict:
    id2filename = {key: '' for key in image_ids}
    for image_id in id_list:
        for element in annotations['images']:
            if image_id == element['id']:
                id2filename[image_id] = element['file_name']
    return id2filename


def reformat_box_to_coco(box: torch.Tensor):
    box = [int(i) for i in box.tolist()]
    formatted_box = [
        box[0],
        box[1],
        box[2] - box[0],
        box[3] - box[1]
    ]
    return formatted_box


def create_rectangle_list(results, dataset_labels, model):
    rectangle_list = []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        formatted_box = reformat_box_to_coco(box)
        score = round(score.item(), 3)
        detected_label = dataset_labels[model.config.id2label[label.item()]]

        formatted_box.append(detected_label)
        formatted_box.append(float(score))

        rectangle_list.append(tuple(formatted_box))

    return rectangle_list


def send_rectangle_list_to_evaluations(rectangle_list, experiment, asset):
    if len(rectangle_list) > 0:
        try:
            experiment.add_evaluation(asset=asset, rectangles=rectangle_list)
            print(f"Asset: {asset.filename} evaluated.")
        except Exception as e:
            print(e)
            pass


def find_asset_from_path(image_path: str, dataset: DatasetVersion) -> Asset:
    asset_filename = get_filename_from_fullpath(image_path)
    try:
        asset = dataset.find_asset(filename=asset_filename)
    except Exception as e:
        print(e)
    return asset


def get_filename_from_fullpath(full_path: str) -> str:
    return full_path.split("/")[-1]


def evaluate_asset(file_path: str, data_dir: str, experiment: Experiment, dataset_labels, model):
    image_path = os.path.join(data_dir, file_path)
    asset = find_asset_from_path(image_path=image_path)
    results = predict_image(image_path=image_path, threshold=0.5)
    rectangle_list = create_rectangle_list(results, dataset_labels, model)
    send_rectangle_list_to_evaluations(rectangle_list, experiment, asset)
