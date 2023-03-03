import json
import logging
import os

from picsellia.types.enums import AnnotationFileType
from pycocotools.coco import COCO

import picsellia_utils
import yolo_utils
from picsellia_detection_trainer import PicselliaDetectionTrainer

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True"

logging.getLogger('picsellia').setLevel(logging.INFO)

experiment = picsellia_utils.get_experiment()

experiment.download_artifacts(with_tree=True)
current_dir = os.path.join(os.getcwd(), experiment.base_dir)
base_imgdir = experiment.png_dir
parameters = experiment.get_log(name="parameters").data

if len(experiment.list_attached_dataset_versions()) == 3:
    train_ds, val_ds, test_ds = (
        experiment.get_dataset(name="train"),
        experiment.get_dataset(name="val"),
        experiment.get_dataset(name="test"),
    )

    for data_type, dataset in {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }.items():
        annotation_path = dataset.export_annotation_file(
            AnnotationFileType.COCO, current_dir
        )
        f = open(annotation_path)
        annotations_dict = json.load(f)
        annotations_coco = COCO(annotation_path)

        if data_type == "train":
            labelmap = {}
            for x in annotations_dict["categories"]:
                labelmap[str(x["id"])] = x["name"]

        dataset.list_assets().download(
            target_path=os.path.join(base_imgdir, data_type, "images"), max_workers=8
        )
        picsellia_utils.create_yolo_detection_label(
            experiment, data_type, annotations_dict, annotations_coco
        )

else:
    dataset = experiment.list_attached_dataset_versions()[0]

    annotation_path = dataset.export_annotation_file(
        AnnotationFileType.COCO, current_dir
    )
    f = open(annotation_path)
    annotations_dict = json.load(f)
    annotations_coco = COCO(annotation_path)
    labelmap = {}
    for x in annotations_dict["categories"]:
        labelmap[str(x["id"])] = x["name"]

    prop = (
        0.7
        if not "prop_train_split" in parameters.keys()
        else parameters["prop_train_split"]
    )

    train_assets, test_assets, val_assets = picsellia_utils.train_test_val_split(
        experiment, dataset, prop, len(annotations_dict["images"])
    )

    for data_type, assets in {
        "train": train_assets,
        "val": val_assets,
        "test": test_assets,
    }.items():
        assets.download(
            target_path=os.path.join(base_imgdir, data_type, "images"), max_workers=8
        )
        picsellia_utils.create_yolo_detection_label(
            experiment, data_type, annotations_dict, annotations_coco
        )

experiment.log("labelmap", labelmap, "labelmap", replace=True)
cwd = os.getcwd()
data_yaml_path = picsellia_utils.generate_data_yaml(experiment, labelmap, current_dir)
cfg = yolo_utils.setup_hyp(
    experiment=experiment,
    data_yaml_path=data_yaml_path,
    params=parameters,
    label_map=labelmap,
    cwd=current_dir,
    task='detect'
)

trainer = PicselliaDetectionTrainer(experiment=experiment, cfg=cfg)
trainer.train()

picsellia_utils.send_run_to_picsellia(experiment, current_dir, trainer.save_dir)