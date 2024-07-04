import yaml

from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext
from src.models.model.model_context import ModelContext
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from src.models.parameters.common.paddle_ocr.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)


def generate_bbox_yaml_config(
    dataset_collection: DatasetCollection[PaddleOCRDatasetContext],
    model_context: ModelContext,
    hyperparameters: PaddleOCRHyperParameters,
):
    if model_context.config_file_path is None:
        raise ValueError("No config file path provided for the model context")

    with open(model_context.config_file_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["Global"]["use_gpu"] = False
    config["Global"]["epoch_num"] = hyperparameters.bbox_epochs
    config["Global"]["pretrained_model"] = model_context.pretrained_model_path
    config["Global"]["save_model_dir"] = model_context.trained_model_path
    config["Global"]["save_res_path"] = model_context.results_path
    config["Global"]["save_inference_dir"] = model_context.inference_model_path
    config["Global"]["save_epoch_step"] = hyperparameters.bbox_save_epoch_step

    config["Optimizer"]["lr"]["learning_rate"] = hyperparameters.bbox_learning_rate

    config["Train"]["dataset"]["name"] = "SimpleDataSet"
    config["Train"]["dataset"]["data_dir"] = dataset_collection.train.image_dir
    config["Train"]["dataset"]["label_file_list"] = [
        dataset_collection.train.paddle_ocr_annotations_path
    ]
    config["Train"]["loader"]["batch_size_per_card"] = hyperparameters.bbox_batch_size
    config["Train"]["loader"]["shuffle"] = True

    config["Eval"]["dataset"]["name"] = "SimpleDataSet"
    config["Eval"]["dataset"]["data_dir"] = dataset_collection.val.image_dir
    config["Eval"]["dataset"]["label_file_list"] = [
        dataset_collection.val.paddle_ocr_annotations_path
    ]
    config["Eval"]["loader"]["batch_size_per_card"] = hyperparameters.bbox_batch_size
    config["Eval"]["loader"]["shuffle"] = True

    return config


def generate_text_yaml_config(
    dataset_collection: DatasetCollection[PaddleOCRDatasetContext],
    model_context: ModelContext,
    hyperparameters: PaddleOCRHyperParameters,
):
    if model_context.config_file_path is None:
        raise ValueError("No config file path provided for the model context")

    with open(model_context.config_file_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["Global"]["use_gpu"] = False
    config["Global"]["epoch_num"] = hyperparameters.text_epochs
    config["Global"]["pretrained_model"] = model_context.pretrained_model_path
    config["Global"]["save_model_dir"] = model_context.trained_model_path
    config["Global"]["save_res_path"] = model_context.results_path
    config["Global"]["save_inference_dir"] = model_context.inference_model_path
    config["Global"]["save_epoch_step"] = hyperparameters.text_save_epoch_step

    config["Optimizer"]["lr"]["learning_rate"] = hyperparameters.text_learning_rate

    config["Train"]["dataset"]["name"] = "SimpleDataSet"
    config["Train"]["dataset"]["data_dir"] = dataset_collection.train.image_dir
    config["Train"]["dataset"]["label_file_list"] = [
        dataset_collection.train.paddle_ocr_annotations_path
    ]
    config["Train"]["loader"]["batch_size_per_card"] = hyperparameters.text_batch_size
    config["Train"]["loader"]["shuffle"] = True

    config["Eval"]["dataset"]["name"] = "SimpleDataSet"
    config["Eval"]["dataset"]["data_dir"] = dataset_collection.val.image_dir
    config["Eval"]["dataset"]["label_file_list"] = [
        dataset_collection.val.paddle_ocr_annotations_path
    ]
    config["Eval"]["loader"]["batch_size_per_card"] = hyperparameters.text_batch_size
    config["Eval"]["loader"]["shuffle"] = True

    return config


def write_yaml_config(config, config_file_path):
    with open(config_file_path, "w") as file:
        yaml.dump(config, file)


class PaddleOCRModelCollectionPreparator:
    def __init__(
        self,
        model_collection: PaddleOCRModelCollection,
        dataset_collection: DatasetCollection,
        hyperparameters: PaddleOCRHyperParameters,
    ):
        self.model_collection = model_collection
        self.dataset_collection = dataset_collection
        self.destination_path = model_collection.bbox_model.destination_path
        self.hyperparameters = hyperparameters

    def prepare(self):
        bbox_config = generate_bbox_yaml_config(
            self.dataset_collection,
            self.model_collection.bbox_model,
            self.hyperparameters,
        )
        text_config = generate_text_yaml_config(
            self.dataset_collection,
            self.model_collection.text_model,
            self.hyperparameters,
        )
        write_yaml_config(
            bbox_config, self.model_collection.bbox_model.config_file_path
        )
        write_yaml_config(
            text_config, self.model_collection.text_model.config_file_path
        )
        return self.model_collection
