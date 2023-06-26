from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from pycocotools.coco import COCO
from utils import _move_files_in_class_directories, get_experiment, prepare_datasets_with_annotation
from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import shutil
from picsellia.exceptions import ResourceNotFoundError
from evaluator.yolo_evaluator import ClassificationYOLOEvaluator

experiment = get_experiment()

dataset_list = experiment.list_attached_dataset_versions()

if len(dataset_list) == 3:
    try:
        train_set = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', 'eval')")
    try:
        test_set = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                            expecting 'train', 'test', 'eval')")
    try:
        val_set = experiment.get_dataset(name="val")
    except Exception:
        try:
            val_set = experiment.get_dataset(name="eval")
        except Exception:
            raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                expecting 'train', 'test', 'eval')")

    for data_type, dataset in {'train': train_set, 'test': test_set, 'val': val_set}.items():
        dataset.download(
                target_path=os.path.join("data", data_type), max_workers=8
            )

    evaluation_ds, evaluation_assets = prepare_datasets_with_annotation(train_set, test_set, val_set)

elif len(dataset_list) == 2:
    train_set = experiment.get_dataset("train")
    train_set.download("data/train")
    test_set = experiment.get_dataset("test")
    test_set.download("data/test")
    val_set = experiment.get_dataset("test")
    val_set.download("data/val")

    # labels = train_set.list_labels()
    # label_names = [label.name for label in labels]
    # labelmap = {str(i): label.name for i, label in enumerate(labels)}
    # experiment.log("labelmap", labelmap, "labelmap", replace=True)

    evaluation_ds, evaluation_assets = prepare_datasets_with_annotation(train_set, test_set, val_set)

elif len(dataset_list) == 1:
    train_set = dataset_list[0]
    train_set.download("images")

    # labels = train_set.list_labels()
    # label_names = [label.name for label in labels]
    # labelmap = {str(i): label.name for i, label in enumerate(labels)}
    # experiment.log("labelmap", labelmap, "labelmap", replace=True)
    parameters = experiment.get_log("parameters").data
    prop = (
        0.7
        if not "prop_train_split" in parameters.keys()
        else parameters["prop_train_split"]
    )
    train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)
    train_assets, test_assets, eval_assets, train_rep, test_rep, val_rep, labels = train_set.train_test_val_split(
        [prop, (1 - prop) / 2, (1 - prop) / 2])
    experiment.log('train-split', train_rep, 'bar', replace=True)
    experiment.log('test-split', test_rep, 'bar', replace=True)
    experiment.log('val-split', val_rep, 'bar', replace=True)
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    for asset in train_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join("data/train", asset.filename)
        shutil.move(old_path, new_path)

    for asset in test_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join("data/test", asset.filename)
        shutil.move(old_path, new_path)

    for asset in eval_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join("data/val", asset.filename)
        shutil.move(old_path, new_path)
    _move_files_in_class_directories(coco_train, "data/train")
    _move_files_in_class_directories(coco_train, "data/test")
    _move_files_in_class_directories(coco_train, "data/val")

    evaluation_ds = train_set
    evaluation_assets = eval_assets
else:
    raise Exception("You must either have only one Dataset, 2 (train, test) or 3 datasets (train, test, eval)")

names = os.listdir("data/train")  # class names list
labelmap = {str(i): label for i, label in enumerate(sorted(names))}
experiment.log("labelmap", labelmap, "labelmap", replace=True)

weights_artifact = experiment.get_artifact("weights")
weights_artifact.download()

model = YOLO(weights_artifact.filename)

working_dir = os.getcwd()
data_path = os.path.join(working_dir, "data")


def on_train_epoch_end(trainer):
    metrics = trainer.metrics
    experiment.log("accuracy", float(metrics["metrics/accuracy_top1"]), "line")
    experiment.log("val/loss", float(metrics["val/loss"]), "line")


model.add_callback("on_train_epoch_end", on_train_epoch_end)
parameters = experiment.get_log("parameters").data
model.train(data=data_path, epochs=parameters["epochs"], imgsz=parameters["image_size"],
            patience=parameters["patience"])

weights_dir_path = os.path.join(working_dir, "runs", "classify", "train", "weights")
weights_path = os.path.join(weights_dir_path, "last.pt")
experiment.store("weights", weights_path)
export_model = YOLO(weights_path)
onnx_path = os.path.join(weights_dir_path, "last.onnx")
export_model.export(format="onnx")
experiment.store("model-latest", onnx_path)

metrics = model.val(data=data_path)
accuracy = metrics.top1
experiment.log("val/accuracy", float(accuracy), "value")

val_folder_path = os.path.join(data_path, "val")
gt_class = []
pred_class = []
for class_id, label in labelmap.items():
    label_path = os.path.join(val_folder_path, label)
    if os.path.exists(label_path):
        file_list = [os.path.join(label_path, filepath) for filepath in os.listdir(label_path)]
        for image in file_list:
            pred = model.predict(source=image)
            pred_label = np.argmax([float(score) for score in list(pred[0].probs)])
            gt_class.append(int(class_id))
            pred_class.append(pred_label)

classification_report(gt_class, pred_class, target_names=labelmap.values())
matrix = confusion_matrix(gt_class, pred_class)

confusion = {"categories": list(labelmap.values()), "values": matrix.tolist()}

experiment.log(name='confusion', data=confusion, type="heatmap")

X = ClassificationYOLOEvaluator(
    experiment=experiment,
    dataset=evaluation_ds,
    asset_list=evaluation_assets,
    confidence_threshold=parameters.get("confidence_threshold", 0.1)
)

X.evaluate()
