from picsellia import Client
from uuid import uuid4
from picsellia.exceptions import (
    ResourceNotFoundError,
    InsufficientResourcesError,
    PicselliaError,
)
from picsellia.types.enums import InferenceType
from picsellia.sdk.model import ModelVersion
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.annotation import Annotation
from picsellia.sdk.label import Label
from typing import List, Tuple, Optional
import tqdm
import zipfile
import os
from formatter import TensorflowFormatter
from PIL import Image
import numpy as np
import requests
import logging
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


class Evaluator:
    """ """

    model = None
    labelmap = {}
    labels_to_detect = None
    dataset_labels = {}

    def __init__(
        self,
        client: Client,
        dataset: DatasetVersion,
        experiment: Experiment,
        parameters: Optional[dict] = None,
        asset_list: List[Asset] = None,
    ) -> None:

        self.client = client
        self.experiment = experiment
        self.dataset_object = dataset
        self.parameters = parameters if parameters is not None else {}
        self.asset_list = asset_list

    def preannotate(self, confidence_threshold: float = 0.1):
        if "batch_size" not in self.parameters:
            batch_size = 8
        else:
            batch_size = self.parameters["batch_size"]
        if self.asset_list:
            asset_list = self.asset_list
        else:
            asset_list = self.dataset_object.list_assets()
            
        confidence_threshold = self.parameters.get("confidence_threshold", confidence_threshold)
        batch_size = batch_size if len(asset_list) > batch_size else len(asset_list)
        self._evaluate_asset_list(asset_list, batch_size, confidence_threshold)
        self.experiment.compute_evaluations_metrics(
            inference_type=self.dataset_object.type
        )

    def _evaluate_asset_list(
        self,
        asset_list: List[Asset],
        batch_size: int,
        confidence_threshold: float = 0.1,
    ):
        total_batch_number = len(asset_list) // batch_size
        for _ in tqdm.tqdm(range(total_batch_number)):
            for asset in asset_list:
                image = self._preprocess_image(asset)
                try:
                    predictions = self.model(image)  # Predict
                except Exception as e:
                    print(e)
                    self.model = tf.saved_model.load(self.model_weights_path)
                    predictions = self.model(image)
                if len(predictions) > 0:
                    #  Format the raw output
                    if self.dataset_object.type == InferenceType.OBJECT_DETECTION:
                        self._format_and_add_rectangles_evaluation(asset, predictions, confidence_threshold)
                    elif self.dataset_object.type == InferenceType.SEGMENTATION:
                        self._format_and_add_polygons_evaluation(
                            asset, predictions, confidence_threshold
                        )

    def get_image_shape_with_exif_transpose(self, image: Image):
        """
            This method reads exif tags of an image and invert width and height if needed.
            Orientation flags that need inversion are : TRANSPOSE, ROTATE_90, TRANSVERSE and ROTATE_270

        Args:
            image: PIL Image to read

        Returns:
            width and height of image
        """
        exif = image.getexif()
        orientation = exif.get(0x0112)

        # Orientation when height and width are inverted :
        # 5: Image.Transpose.TRANSPOSE
        # 6: Image.Transpose.ROTATE_270
        # 7: Image.Transpose.TRANSVERSE
        # 8: Image.Transpose.ROTATE_90
        if orientation == 3:
            image=image.rotate(180, expand=True)
        elif orientation == 6:
            image=image.rotate(270, expand=True)
        elif orientation == 8:
            image=image.rotate(90, expand=True)
        return image

    def setup_preannotation_job(
        self,
    ):
        logging.info(f"Setting up the evaluation for this experiment")
        self.model_sanity_check()
        self._dataset_inclusion_check()

        self._labels_coherence_check()
        self.labels_to_detect = list(
            set(self.model_labels_name).intersection(self.dataset_labels_name)
        )
        self._download_model_weights()
        self._load_tensorflow_saved_model()

    # Coherence Checks

    def _labels_coherence_check(self) -> bool:
        """
        Assert that at least one label from the model labelmap is contained in the dataset version.
        """
        self.model_labels_name = list(self._get_experiment_labelmap().values())
        labels = self.dataset_object.list_labels()
        self.dataset_labels_name = [label.name for label in labels]
        self.dataset_labels = {label.name: label for label in labels}
        intersecting_labels = set(self.model_labels_name).intersection(
            self.dataset_labels_name
        )
        logging.info(
            f"Pre-annotation Job will only run on classes: {list(intersecting_labels)}"
        )
        return len(intersecting_labels) > 0

    def _dataset_inclusion_check(
        self,
    ) -> None:
        """
        Check if the selected dataset is included into the given experiment,

        If the dataset isn't in the experiment, we'll add it under the name "eval".
        """

        attached_datasets = self.experiment.list_attached_dataset_versions()
        inclusion = False
        for dataset_version in attached_datasets:
            if dataset_version.id == self.dataset_object.id:
                inclusion = True

        if not inclusion:
            self.experiment.attach_dataset(
                name="eval", dataset_version=self.dataset_object
            )
            logging.info(
                f"{self.dataset_object.name}/{self.dataset_object.version} attached to the experiment."
            )
        return

    # Sanity check

    def _check_model_file_sanity(
        self,
    ) -> None:
        try:
            self.experiment.get_artifact("model-latest")
        except ResourceNotFoundError as e:
            raise ResourceNotFoundError(
                f"Can't run a pre-annotation job with this model, expected a 'model-latest' file"
            )

    def model_sanity_check(
        self,
    ) -> None:
        self._check_model_file_sanity()
        logging.info(f"Experiment {self.experiment.name} is sane.")

    # Utilities

    def _is_labelmap_starting_at_zero(
        self,
    ) -> bool:
        return "0" in self.labelmap.keys()

    def _get_experiment_labelmap(
        self,
    ) -> dict:
        try:
            self.labelmap = self.experiment.get_log("labelmap").data
        except Exception:
            raise InsufficientResourcesError(f"Can't find labelmap for this experiment")
        return self.labelmap

    def _download_model_weights(
        self,
    ):
        model_weights = self.experiment.get_artifact("model-latest")
        model_weights.download()
        weights_zip_path = model_weights.filename
        with zipfile.ZipFile(weights_zip_path, "r") as zip_ref:
            zip_ref.extractall("saved_model")
        cwd = os.getcwd()
        self.model_weights_path = os.path.join(cwd, "saved_model")
        logging.info(f"experiment weights downloaded.")

    def _load_tensorflow_saved_model(
        self,
    ):
        try:
            self.model = tf.saved_model.load(self.model_weights_path)
            logging.info("Model loaded in memory.")
            try:
                self.model = self.model.signatures[
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                ]
                self.input_width, self.input_height = (
                    self.model.inputs[0].shape[1],
                    self.model.inputs[0].shape[2],
                )
                self.output_names = list(self.model.structured_outputs.keys())
            except Exception:
                self.input_width, self.input_height = None, None
                self.output_names = None
        except Exception as e:
            raise PicselliaError(
                f"Impossible to load saved model located at: {self.model_weights_path}"
            )

    def _preprocess_image(self, asset: Asset) -> np.array:
        image = Image.open(
            requests.get(asset.sync()["data"]["presigned_url"], stream=True).raw
        )
        image = self.get_image_shape_with_exif_transpose(image)
        if self.input_width is not None and self.input_height is not None:
            image = image.resize((self.input_width, self.input_height))
            if image.mode != "RGB":
                image = image.convert("RGB")
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        if self.input_width is not None and self.input_height is not None:
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        return image

    def _format_picsellia_rectangles(
        self, width: int, height: int, predictions: np.array
    ) -> Tuple[List, List, List]:
        formatter = TensorflowFormatter(width, height, self.output_names)
        formated_output = formatter.format_object_detection(predictions)
        scores = formated_output["detection_scores"]
        boxes = formated_output["detection_boxes"]
        classes = formated_output["detection_classes"]
        return scores, boxes, classes

    def _format_picsellia_polygons(
        self, width: int, height: int, predictions: np.array
    ) -> Tuple[List, List, List, List]:
        formatter = TensorflowFormatter(width, height, self.output_names)
        formated_output = formatter.format_segmentation(predictions)
        scores = formated_output["detection_scores"]
        boxes = formated_output["detection_boxes"]
        classes = formated_output["detection_classes"]
        masks = formated_output["detection_masks"]
        return scores, masks, boxes, classes

    def _format_and_add_rectangles_evaluation(
        self, asset: Asset, predictions: dict, confidence_threshold: float = 0.1
    ) -> None:
        scores, boxes, classes = self._format_picsellia_rectangles(
            width=asset.width, height=asset.height, predictions=predictions
        )
        #  Convert predictions to Picsellia format
        rectangle_list = []
        nb_box_limit = 100
        if len(boxes) < nb_box_limit:
            nb_box_limit = len(boxes)
        if len(boxes) == 0:
            return
        # print(scores, boxes, classes)
        for i in range(nb_box_limit):
            if scores[i] >= confidence_threshold:
                if str(int(classes[i])) in self.labelmap.keys():
                    try:
                        label: Label = self.dataset_labels[
                            self.labelmap[str(int(classes[i]))]
                        ]

                        box = boxes[i]
                        box.append(label)
                        box.append(scores[i])
                        rectangle_list.append(tuple(box))
                    except ResourceNotFoundError as e:
                        print(e)
                        continue
        if len(rectangle_list) > 0:
            self.experiment.add_evaluation(asset=asset, rectangles=rectangle_list)
            logging.info(f"Asset: {asset.filename} evaluated.")

    def _format_and_add_polygons_evaluation(
        self, asset: Asset, predictions: dict, confidence_threshold: float
    ) -> None:
        scores, masks, _, classes = self._format_picsellia_polygons(
            width=asset.width, height=asset.height, predictions=predictions
        )
        #  Convert predictions to Picsellia format
        polygons_list = []
        nb_polygons_limit = 100
        if len(masks) < nb_polygons_limit:
            nb_box_limit = len(masks)
        for i in range(nb_box_limit):
            if scores[i] >= confidence_threshold:
                try:
                    if self._is_labelmap_starting_at_zero():
                        label: Label = self.dataset_object.get_label(
                            name=self.labelmap[str(int(classes[i]) - 1)]
                        )
                    else:
                        label: Label = self.dataset_object.get_label(
                            name=self.labelmap[str(int(classes[i]))]
                        )
                    polygons_list.append((masks[i], label, scores[i]))
                except ResourceNotFoundError as e:
                    print(e)
                    continue
        if len(polygons_list) > 0:
            self.experiment.add_evaluation(asset=asset, polygons=polygons_list)
            logging.info(f"Asset: {asset.filename} evaluated.")

        #  Fetch original annotation and shapes to overlay over predictions
