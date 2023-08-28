import os
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

from trainer.abstract_trainer import AbstractTrainer
from utils import (
    get_train_test_eval_datasets_from_experiment,
    download_triple_dataset,
    prepare_datasets_with_annotation,
    split_single_dataset,
    _move_all_files_in_class_directories,
    log_split_dataset_repartition_to_experiment,
    predict_class,
    log_confusion_to_experiment,
)
from evaluator.yolo_evaluator import ClassificationYOLOEvaluator


class Yolov8ClassificationTrainer(AbstractTrainer):
    def __init__(self):
        super().__init__()
        self.weights = self.experiment.get_artifact("weights")
        self.weights.download()
        self.model = YOLO(self.weights.filename)
        self.working_directory = os.getcwd()
        self.data_path = os.path.join(self.working_directory, "data")
        self.val_folder_path = os.path.join(self.data_path, "val")
        self.weights_dir_path = os.path.join(
            self.working_directory, "runs", "classify", "train", "weights"
        )
        self.weights_path = os.path.join(self.weights_dir_path, "last.pt")
        self.evaluation_ds, self.evaluation_assets = None, None

    def prepare_data_for_training(self):
        (
            is_split_two,
            is_split_three,
            train_set,
            test_set,
            eval_set,
        ) = get_train_test_eval_datasets_from_experiment(experiment=self.experiment)
        if is_split_three:
            download_triple_dataset(train_set, test_set, eval_set)
            _, _ = prepare_datasets_with_annotation(train_set, test_set, eval_set)
            self.evaluation_ds = eval_set
            self.evaluation_assets = self.evaluation_ds.list_assets()

        elif is_split_two:
            download_triple_dataset(train_set, test_set, eval_set)
            (
                self.evaluation_ds,
                self.evaluation_assets,
            ) = prepare_datasets_with_annotation(train_set, test_set, eval_set)

        elif not is_split_two and not is_split_three:
            train_set.download("images")
            (
                train_assets,
                test_assets,
                eval_assets,
                train_rep,
                test_rep,
                val_rep,
                labels,
            ) = split_single_dataset(experiment=self.experiment, train_set=train_set)
            _move_all_files_in_class_directories(train_set=train_set)
            self.labelmap = log_split_dataset_repartition_to_experiment(
                train_rep, test_rep, val_rep
            )
            self.evaluation_ds = train_set
            self.evaluation_assets = eval_assets
        else:
            raise Exception(
                "You must either have only one Dataset, 2 (train, test) or 3 datasets (train, test, eval)"
            )

    def train(self):
        self.model.train(
            data=self.data_path,
            epochs=self.parameters["epochs"],
            imgsz=self.parameters["image_size"],
            patience=self.parameters["patience"],
        )
        self.experiment.store("weights", self.weights_path)
        export_model = YOLO(self.weights_path)
        onnx_path = os.path.join(self.weights_dir_path, "last.onnx")
        export_model.export(format="onnx")
        self.experiment.store("model-latest", onnx_path)

    def test(self):
        self.model = YOLO(self.weights_path)
        metrics = self.model.val(data=self.data_path)
        accuracy = metrics.top1
        self.experiment.log("val/accuracy", float(accuracy), "value")

        gt_class, pred_class = predict_class(
            labelmap=self.labelmap,
            val_folder_path=self.val_folder_path,
            model=self.model,
        )
        classification_report(gt_class, pred_class, target_names=self.labelmap.values())
        matrix = confusion_matrix(gt_class, pred_class)
        log_confusion_to_experiment(
            experiment=self.experiment, labelmap=self.labelmap, matrix=matrix
        )

    def eval(self):
        yolo_evaluator = ClassificationYOLOEvaluator(
            experiment=self.experiment,
            dataset=self.evaluation_ds,
            asset_list=self.evaluation_assets,
            confidence_threshold=self.parameters.get("confidence_threshold", 0.1),
            weights_path=self.weights_path,
        )

        yolo_evaluator.evaluate()
