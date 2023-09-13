import json
import logging
import os

from picsellia.exceptions import ResourceNotFoundError
from pycocotools.coco import COCO

from evaluator.yolo_evaluator import SegmentationYOLOEvaluator
from utils import processing, yolo, segmentation_trainer

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"

logging.getLogger("experiment").setLevel(logging.INFO)

experiment = processing.get_experiment()

experiment.download_artifacts(with_tree=True)
current_dir = os.path.join(os.getcwd(), experiment.base_dir)
base_imgdir = experiment.png_dir

parameters = experiment.get_log(name="parameters").data
attached_datasets = experiment.list_attached_dataset_versions()

if len(attached_datasets) == 3:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')"
        )
    try:
        test_ds = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')"
        )
    try:
        val_ds = experiment.get_dataset(name="val")
    except Exception:
        try:
            val_ds = experiment.get_dataset(name="eval")
        except Exception:
            raise ResourceNotFoundError(
                "Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                expecting 'train', 'test', ('val' or 'eval')"
            )
    labels = train_ds.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i): label.name for i, label in enumerate(labels)}

    for data_type, dataset in {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }.items():
        coco_annotation = dataset.build_coco_file_locally(
            enforced_ordered_categories=label_names
        )
        annotations_dict = coco_annotation.dict()
        categories_dict = [
            category["name"] for category in annotations_dict["categories"]
        ]
        for label in label_names:
            if label not in categories_dict:
                annotations_dict["categories"].append(
                    {
                        "id": len(annotations_dict["categories"]),
                        "name": label,
                        "supercategory": "",
                    }
                )
        annotations_path = "annotations.json"
        with open(annotations_path, "w") as f:
            f.write(json.dumps(annotations_dict))
        annotations_coco = COCO(annotations_path)

        dataset.list_assets().download(
            target_path=os.path.join(base_imgdir, data_type, "images"), max_workers=8
        )
        processing.create_yolo_segmentation_label(
            experiment, data_type, annotations_dict, annotations_coco, label_names
        )

    evaluation_ds = test_ds
    evaluation_assets = evaluation_ds.list_assets()

else:
    dataset = experiment.list_attached_dataset_versions()[0]

    labels = dataset.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i): label.name for i, label in enumerate(labels)}

    coco_annotation = dataset.build_coco_file_locally(
        enforced_ordered_categories=label_names
    )
    annotations_dict = coco_annotation.dict()
    categories_dict = [category["name"] for category in annotations_dict["categories"]]
    for label in label_names:
        if label not in categories_dict:
            annotations_dict["categories"].append(
                {
                    "id": len(annotations_dict["categories"]),
                    "name": label,
                    "supercategory": "",
                }
            )
    annotations_path = "annotations.json"
    with open(annotations_path, "w") as f:
        f.write(json.dumps(annotations_dict))
    annotations_coco = COCO(annotations_path)

    prop = (
        0.7
        if not "prop_train_split" in parameters.keys()
        else parameters["prop_train_split"]
    )

    train_assets, test_assets, val_assets = processing.train_test_val_split(
        experiment, dataset, prop, len(annotations_dict["images"]), label_names
    )

    for data_type, assets in {
        "train": train_assets,
        "val": val_assets,
        "test": test_assets,
    }.items():
        assets.download(
            target_path=os.path.join(base_imgdir, data_type, "images"), max_workers=8
        )
        processing.create_yolo_segmentation_label(
            experiment, data_type, annotations_dict, annotations_coco, label_names
        )

    evaluation_ds = dataset
    evaluation_assets = test_assets

experiment.log("labelmap", labelmap, "labelmap", replace=True)
data_yaml_path = processing.generate_data_yaml(experiment, labelmap, current_dir)

cfg = yolo.setup_hyp(
    experiment=experiment,
    data_yaml_path=data_yaml_path,
    params=parameters,
    label_map=labelmap,
    cwd=current_dir,
    task="segment",
)
print(cfg.task)
trainer = segmentation_trainer.PicselliaSegmentationTrainer(
    experiment=experiment, cfg=cfg
)
trainer.train()

processing.send_run_to_picsellia(experiment, current_dir, trainer.save_dir)

X = SegmentationYOLOEvaluator(
    experiment=experiment,
    dataset=evaluation_ds,
    asset_list=evaluation_assets,
    confidence_threshold=parameters.get("confidence_threshold", 0.1),
)

X.evaluate()
