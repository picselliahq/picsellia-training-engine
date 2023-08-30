import os
import unittest
from utils import (
    get_prop_parameter,
    make_train_test_val_dirs,
    move_image,
    _get_three_attached_datasets,
    _create_coco_objects,
    _create_class_directories,
    _move_files_in_class_directories,
    get_image_annotation,
)
from picsellia import Client
from datetime import date
import time

TOKEN = os.environ["TEST_TOKEN"]
ORGA_NAME = os.environ["TEST_ORGA"]


class TestYoloClassification(unittest.TestCase):
    eval_set = None
    test_set = None
    train_set = None
    dataset = None
    experiment = None
    project = None
    client = None
    organization_name: str
    token: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.token = TOKEN
        cls.organization_name = ORGA_NAME
        cls.client = Client(
            api_token=cls.token, organization_name=cls.organization_name
        )
        cls.project = cls.client.create_project(
            name=f"test_yolo_classif{str(date.today())}-{str(time.time())}"
        )
        cls.model_version = cls.client.get_public_model(
            "yolov8-classification"
        ).get_version(0)
        cls.experiment = cls.project.create_experiment(name="yolo-triple-dataset")
        cls.dataset = cls.client.get_dataset_by_id(
            "01888c1b-cfb6-768b-83dd-2e1c460e79cf"
        )
        cls.experiment.attach_dataset(
            name="train", dataset_version=cls.dataset.get_version("train")
        )
        cls.experiment.attach_dataset(
            name="test", dataset_version=cls.dataset.get_version("test")
        )
        cls.experiment.attach_dataset(
            name="eval", dataset_version=cls.dataset.get_version("val")
        )
        cls.train_set, cls.test_set, cls.eval_set = _get_three_attached_datasets(
            cls.experiment
        )
        cls.coco_train, cls.coco_test, cls.coco_val = _create_coco_objects(
            cls.train_set, cls.test_set, cls.eval_set
        )
        cls.train_path = "data/train"
        cls.test_path = "data/test"
        cls.val_path = "data/val"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()

    def test_get_prop_parameters(self):
        parameters = {"epochs": 100, "patience": 500, "image_size": 640}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.7)

    def test_get_prop_parameters_exit(self):
        parameters = {"epochs": 100, "patience": 500, "prop_train_split": 0.8}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.8)

    def test_make_train_test_val_dirs(self):
        make_train_test_val_dirs()
        is_created = False
        if (
            os.path.exists(self.train_path)
            and os.path.exists(self.test_path)
            and os.path.exists(self.val_path)
        ):
            is_created = True
        self.assertTrue(is_created)

    def test_move_image(self):
        move_image(
            filename="grape_image.jpeg",
            old_location_path="test_files",
            new_location_path=self.test_path,
        )
        is_moved = False
        if os.path.isfile(os.path.join(self.test_path, "grape_image.jpeg")):
            is_moved = True
        self.assertTrue(is_moved)

    def test_create_class_directories(self):
        _create_class_directories(coco=self.coco_train, base_imdir=self.train_path)
        is_created = False
        if (
            os.path.exists(os.path.join(self.train_path, "covid"))
            and os.path.exists(os.path.join(self.train_path, "normal"))
            and os.path.exists(os.path.join(self.train_path, "pneumonia"))
        ):
            is_created = True
        self.assertTrue(is_created)

    def test_get_image_annotation(self):
        fnames = ["covid_082.jpg"]
        image = {
            "id": 0,
            "file_name": "covid_082.jpg",
            "width": 1165,
            "height": 1163,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None,
        }
        cat = get_image_annotation(coco=self.coco_train, fnames=fnames, image=image)
        expected_cat = {"id": 0, "name": "covid", "supercategory": None}

        self.assertEqual(expected_cat, cat)


if __name__ == "__main__":
    unittest.main()
