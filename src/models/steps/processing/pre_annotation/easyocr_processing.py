import os
from typing import List

import easyocr
import numpy as np
from PIL import Image

from src.models.dataset.common.dataset_context import DatasetContext

gpu = True


class EasyOcrProcessing:
    def __init__(self, client, input_dataset_context, language):
        self.client = client
        self.dataset_context: DatasetContext = input_dataset_context

        self.reader = easyocr.Reader(
            lang_list=[language],
            gpu=gpu,
            model_storage_directory="src/pipelines/easyocr/model",
        )
        self.image_width = 0
        self.image_height = 0

    def _crop(self, image: Image, box: List) -> np.array:
        x, y, w, h = box
        self.image = np.array(image.convert("L"))
        return self.image[y : y + h, x : x + w]

    def predict(self, box: List, image: np.array):
        roi = self._crop(image, box)
        results = self.reader.readtext(roi)
        if len(results) == 0:
            return ""
        return results[0][1]

    def prepare_image_map(self):
        image_map = {}
        for image in self.dataset_context.coco_file.images:
            image_map[image.id] = image.file_name
        return image_map

    def process(self):
        images = {
            name: os.path.join(self.dataset_context.image_dir, name)
            for name in os.listdir(self.dataset_context.image_dir)
        }
        image_map = self.prepare_image_map()
        print("Starting OCR processing")
        for i, object in enumerate(self.dataset_context.coco_file.annotations):
            image_id = object.image_id
            filename = image_map[image_id]
            if filename in images.keys():
                image = Image.open(images[filename])
                box = object.bbox
                prediction = self.predict(box, image)
                object.utf8_string = prediction
            if i % 100 == 0:
                print(
                    f"Processed {int((i / len(self.dataset_context.coco_file.annotations)) * 100)}% of the shapes"
                )
        coco_json = self.dataset_context.coco_file.model_dump_json()
        with open("annotations_updated.json", "w") as f:
            f.write(coco_json)
        self.dataset_context.dataset_version.import_annotations_coco_file(
            "annotations_updated.json", use_id=True
        )
