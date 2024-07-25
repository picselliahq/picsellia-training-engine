import json
import os
from typing import Union

from paddleocr import PaddleOCR
from picsellia.types.enums import InferenceType

from src.models.dataset.common.dataset_context import DatasetContext
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from .predict_image import predict_image, get_annotations_from_result


class PaddleOcrProcessing:
    def __init__(
        self,
        client,
        input_dataset_context: DatasetContext,
        model_collection: PaddleOCRModelCollection,
    ):
        self.client = client
        self.dataset_context: DatasetContext = input_dataset_context
        self.image_width = 0
        self.image_height = 0
        self.model_collection = model_collection

    def process(self):
        annotations_path = os.path.join(
            self.dataset_context.destination_path,
            self.dataset_context.dataset_name,
            "annotations_updated.json",
        )
        coco_json = self.dataset_context.coco_file.model_dump_json()
        with open(annotations_path, "w") as f:
            f.write(coco_json)
        self.predict_dataset_version(self.dataset_context.image_dir, annotations_path)
        self.dataset_context.dataset_version.set_type(InferenceType.OBJECT_DETECTION)
        self.dataset_context.dataset_version.import_annotations_coco_file(
            annotations_path, use_id=True
        )

    def predict_dataset_version(self, images_dir, annotations_path):
        ocr_model = self.load_model()
        with open(annotations_path) as f:
            cocotext = json.load(f)
        cocotext["categories"] = [{"id": 0, "name": "text", "supercategory": "text"}]
        for image_filename in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_filename)
            result = predict_image(ocr_model, image_path)
            if result[0]:
                bboxes, texts, scores = get_annotations_from_result(result)
                for bbox, text, score in zip(bboxes, texts, scores):
                    print(f"bbox: {bbox}, text: {text}, score: {score}")
                    coco_bbox = self.oriented_to_aligned_bbox(bbox)
                    print(f"coco_bbox: {coco_bbox}")
                    cocotext["annotations"].append(
                        {
                            "id": len(cocotext["annotations"]),
                            "image_id": self.find_image_id(
                                cocotext["images"], image_filename
                            ),
                            "category_id": 0,
                            "bbox": coco_bbox,
                            "utf8_string": text,
                            "area": coco_bbox[2] * coco_bbox[3],
                            "segmentation": [],
                        }
                    )
        with open(annotations_path, "w") as f:
            json.dump(cocotext, f)

    def load_model(self) -> PaddleOCR:
        print(self.model_collection.text_model.pretrained_model_path)
        print(self.model_collection.bbox_model.pretrained_model_path)
        return PaddleOCR(
            lang="en",
            use_angle_cls=True,
            rec_model_dir=self.model_collection.text_model.pretrained_model_path,
            det_model_dir=self.model_collection.bbox_model.pretrained_model_path,
            rec_char_dict_path=os.path.join(
                self.model_collection.text_model.model_weights_path, "en_dict.txt"
            ),
            use_gpu=False,
            show_log=False,
        )

    @staticmethod
    def download_dataset_version(dataset_version):
        dataset_dir = f"{dataset_version.name}/{dataset_version.version}"
        os.makedirs(dataset_dir, exist_ok=True)
        images_dir = f"{dataset_dir}/images"
        annotations_dir = f"{dataset_dir}/annotations"
        dataset_version.download(target_path=images_dir)
        annotations_path = dataset_version.export_annotation_file(
            "coco", annotations_dir
        )
        return images_dir, annotations_path

    @staticmethod
    def find_image_id(images, image_filename: str) -> Union[str, None]:
        for image in images:
            if image["file_name"] == image_filename:
                print(f'Found image id {image["id"]} for image {image_filename}')
                return image["id"]
        return None

    @staticmethod
    def oriented_to_aligned_bbox(points):
        # Extraire les listes de coordonnées x et y
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # Trouver les minimums et maximums
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Calculer la largeur et la hauteur
        width = max_x - min_x
        height = max_y - min_y

        # Retourner les coordonnées du coin inférieur gauche, la largeur et la hauteur
        return [min_x, min_y, width, height]
