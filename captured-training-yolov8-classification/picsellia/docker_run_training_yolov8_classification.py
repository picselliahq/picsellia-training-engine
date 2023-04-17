from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from pycocotools.coco import COCO
from utils import _move_files_in_class_directories, get_experiment
from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


experiment = get_experiment()

train_set = experiment.get_dataset("train")
train_set.download("data/train")
test_set = experiment.get_dataset("test")
test_set.download("data/test")
val_set = experiment.get_dataset("test")
val_set.download("data/val")

labels = train_set.list_labels()
label_names = [label.name for label in labels]
labelmap = {str(i): label.name for i, label in enumerate(labels)}
experiment.log("labelmap", labelmap, "labelmap", replace=True)

train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
coco_train = COCO(train_annotation_path)

test_annotation_path = test_set.export_annotation_file(AnnotationFileType.COCO)
coco_test = COCO(test_annotation_path)

val_annotation_path = val_set.export_annotation_file(AnnotationFileType.COCO)
coco_val = COCO(val_annotation_path)

_move_files_in_class_directories(coco_train, "data/train")
_move_files_in_class_directories(coco_test, "data/test")
_move_files_in_class_directories(coco_val, "data/val")

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
model.train(data=data_path, epochs=parameters["epochs"], imgsz=parameters["image_size"], patience=parameters["patience"])

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
class_list = os.listdir(val_folder_path)
gt_class = []
pred_class = []
for class_id, label in enumerate(class_list):
    label_path = os.path.join(val_folder_path, label)
    file_list = [os.path.join(label_path, filepath) for filepath in os.listdir(label_path)]
    for image in file_list:
        pred = model.predict(source=image)
        pred_label = np.argmax([float(score) for score in list(pred[0].probs)])
        gt_class.append(class_id)
        pred_class.append(pred_label)

classification_report(gt_class, pred_class, target_names=class_list)
matrix = confusion_matrix(gt_class, pred_class)

confusion = {"categories": class_list, "values": matrix.tolist()}

experiment.log(name='confusion', data=confusion, type="heatmap")