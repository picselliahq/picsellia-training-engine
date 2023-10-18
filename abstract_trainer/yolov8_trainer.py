import os
from abc import abstractmethod

from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.dataset import DatasetVersion
from pycocotools.coco import COCO
from picsellia.types.enums import LogType
from abstract_trainer.trainer import AbstractTrainer
from core_utils.yolov8 import (
    get_train_test_eval_datasets_from_experiment,
    write_annotation_file,
    get_prop_parameter,
    log_split_dataset_repartition_to_experiment,
    make_annotation_dict_by_dataset,
    get_metrics_curves,
    extract_file_name,
    get_batch_mosaics,
)


class Yolov8Trainer(AbstractTrainer):
    def __init__(self):
        super().__init__()
        self.trainer = None
        self.evaluation_ds = None
        self.evaluation_assets = None
        self.annotations_dict = None
        self.annotations_coco = None
        self.label_names = None
        self.current_dir = os.path.join(os.getcwd(), self.experiment.base_dir)
        self.annotations_path = "annotations.json"

    def prepare_data_for_training(self):
        self.experiment.download_artifacts(with_tree=True)
        (
            has_three_datasets,
            train_set,
            test_set,
            eval_set,
        ) = get_train_test_eval_datasets_from_experiment(self.experiment)

        if train_set:
            labels = train_set.list_labels()
            self.label_names = [label.name for label in labels]
            self.labelmap = {str(i): label.name for i, label in enumerate(labels)}
            self.experiment.log("labelmap", self.labelmap, "labelmap", replace=True)
        if has_three_datasets:
            for data_type, dataset in {
                "train": train_set,
                "test": test_set,
                "val": eval_set,
            }.items():
                self._prepare_annotations_label(data_type=data_type, dataset=dataset)
            self.evaluation_ds = test_set
            self.evaluation_assets = self.evaluation_ds.list_assets()
        else:
            self._process_single_dataset(train_set=train_set)

    def _process_single_dataset(self, train_set: DatasetVersion):
        self.annotations_dict = make_annotation_dict_by_dataset(
            dataset=train_set, label_names=self.label_names
        )
        write_annotation_file(
            annotations_dict=self.annotations_dict,
            annotations_path=self.annotations_path,
        )
        self.annotations_coco = COCO(self.annotations_path)

        prop = get_prop_parameter(parameters=self.parameters)
        (
            train_assets,
            test_assets,
            eval_assets,
            train_rep,
            test_rep,
            val_rep,
            labels,
        ) = train_set.train_test_val_split(([prop, (1 - prop) / 2, (1 - prop) / 2]))

        for data_type, assets in {
            "train": train_assets,
            "val": eval_assets,
            "test": test_assets,
        }.items():
            self.download_data_with_label(assets=assets, data_type=data_type)
        self.evaluation_ds = train_set
        self.evaluation_assets = test_assets
        log_split_dataset_repartition_to_experiment(
            experiment=self.experiment,
            labelmap=self.labelmap,
            train_rep=train_rep,
            test_rep=test_rep,
            val_rep=val_rep,
        )

    def _prepare_annotations_label(self, data_type: str, dataset: DatasetVersion):
        self.annotations_dict = make_annotation_dict_by_dataset(
            dataset=dataset, label_names=self.label_names
        )
        write_annotation_file(
            annotations_dict=self.annotations_dict,
            annotations_path=self.annotations_path,
        )
        self.annotations_coco = COCO(self.annotations_path)
        self.download_data_with_label(assets=dataset.list_assets(), data_type=data_type)

    @abstractmethod
    def download_data_with_label(self, assets: MultiAsset, data_type: str):
        pass

    def train(self):
        self.setup_trainer()
        self.launch_trainer()

    @abstractmethod
    def setup_trainer(self):
        pass

    def launch_trainer(self):
        self.trainer.train()
        self.send_run_to_picsellia()

    def send_run_to_picsellia(self):
        final_run_path = self.trainer.save_dir
        for curve in get_metrics_curves(final_run_path):
            if curve:
                name = extract_file_name(curve)
                self.experiment.log(name, curve, LogType.IMAGE)

        for batch in get_batch_mosaics(final_run_path):
            if batch:
                name = extract_file_name(batch)
                self.experiment.log(name, batch, LogType.IMAGE)

    @abstractmethod
    def eval(self):
        pass
