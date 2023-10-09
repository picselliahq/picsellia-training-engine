import os
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
from evaluator.yolo_evaluator import ClassificationYOLOEvaluator
from abstract_trainer.trainer import AbstractTrainer

from utils import (
    get_train_test_eval_datasets_from_experiment,
    download_triple_dataset,
    prepare_datasets_with_annotation,
    split_single_dataset,
    _move_all_files_in_class_directories,
    log_split_dataset_repartition_to_experiment,
    predict_evaluation_images,
    log_confusion_to_experiment,
    create_and_log_labelmap,
)


class Yolov8ClassificationTrainer(AbstractTrainer):
    def __init__(self):
        super().__init__()
        self.weights = self.experiment.get_artifact("weights")
        self.weights.download()
        self.model = YOLO(self.weights.filename)
        self.working_directory = os.getcwd()
        self.data_path = os.path.join(self.working_directory, "data")
        self.val_folder_path = os.path.join(self.data_path, "val")
        self.weights_dir_path = os.path.join(self.working_directory, "train", "weights")
        self.weights_path = os.path.join(self.weights_dir_path, "last.pt")
        self.evaluation_ds, self.evaluation_assets = None, None

    def prepare_data_for_training(self):
        (
            has_three_datasets,
            train_set,
            test_set,
            eval_set,
        ) = get_train_test_eval_datasets_from_experiment(experiment=self.experiment)
        if has_three_datasets:
            download_triple_dataset(train_set, test_set, eval_set)
            _, _ = prepare_datasets_with_annotation(train_set, test_set, eval_set)
            self.evaluation_ds = eval_set
            self.evaluation_assets = self.evaluation_ds.list_assets()

        else:
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
            self.labelmap = create_and_log_labelmap(experiment=self.experiment)
            log_split_dataset_repartition_to_experiment(
                experiment=self.experiment,
                labelmap=self.labelmap,
                train_rep=train_rep,
                test_rep=test_rep,
                val_rep=val_rep,
            )
            self.evaluation_ds = train_set
            self.evaluation_assets = eval_assets
        else:
            raise Exception(
                "You must either have only one Dataset, 2 (train, test) or 3 datasets (train, test, eval)"
            )
        self.labelmap = create_and_log_labelmap(experiment=self.experiment)

    def train(self):
        self.model.train(
            data=self.data_path,
            epochs=self.parameters["epochs"],
            imgsz=self.parameters["image_size"],
            patience=self.parameters["patience"],
            project=self.working_directory,
        )
        self._save_weights()
        self._save_onnx_model()

    def _save_weights(self):
        self.experiment.store("weights", self.weights_path)

    def _save_onnx_model(self):
        export_model = YOLO(self.weights_path)
        onnx_path = os.path.join(self.weights_dir_path, "last.onnx")
        export_model.export(format="onnx")
        self.experiment.store("model-latest", onnx_path)

    def eval(self):
        self.model = YOLO(self.weights_path)
        metrics = self.model.val(data=self.data_path)
        accuracy = metrics.top1
        self.experiment.log("val/accuracy", float(accuracy), "value")

        ground_truths, predictions = predict_evaluation_images(
            labelmap=self.labelmap,
            val_folder_path=self.val_folder_path,
            model=self.model,
        )
        classification_report(
            ground_truths, predictions, target_names=self.labelmap.values()
        )
        matrix = confusion_matrix(ground_truths, predictions)
        log_confusion_to_experiment(
            experiment=self.experiment, labelmap=self.labelmap, matrix=matrix
        )
        self._run_picsellia_evaluation()

    def _run_picsellia_evaluation(self):
        yolo_evaluator = ClassificationYOLOEvaluator(
            experiment=self.experiment,
            dataset=self.evaluation_ds,
            asset_list=self.evaluation_assets,
            confidence_threshold=self.parameters.get("confidence_threshold", 0.1),
            weights_path=self.weights_path,
        )

        yolo_evaluator.evaluate()