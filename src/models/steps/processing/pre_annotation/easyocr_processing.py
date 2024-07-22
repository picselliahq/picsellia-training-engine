import os

import easyocr
import numpy as np
from PIL import Image
import tqdm

from src.models.dataset.common.dataset_context import DatasetContext

gpu = True


class EasyOcrProcessing:

    def __init__(self, client, input_dataset_context, language):
        self.client = client
        self.dataset_context: DatasetContext = input_dataset_context

        self.reader = easyocr.Reader(lang_list=[language], gpu=gpu)
        self.image_width = 0
        self.image_height = 0

    def _crop(self, image: Image, box: list[int, int, int, int]) -> np.array:
        x, y, w, h = box
        self.image = np.array(
            image.convert("L")
        )
        return self.image[y: y + h, x: x + w]

    def predict(self, box: list[int, int, int, int], image: np.array):
        roi = self._crop(image, box)
        results = self.reader.readtext(roi)
        if len(results) == 0:
            return ""
        return results[0][1]

    def process(self):
        images = {name: os.path.join(self.dataset_context.image_dir, name) for name in
                  os.listdir(self.dataset_context.image_dir)}
        print("Starting OCR processing")
        for i, object in enumerate(self.dataset_context.coco_file.annotations):
            image_id = object.image_id
            image = self.dataset_context.coco_file.images[image_id]
            if image.file_name in images.keys():
                image = Image.open(images[image.file_name])
                box = object.bbox
                prediction = self.predict(box, image)
                object.utf8_string = prediction
            if i % 100 and i > 0 == 0:
                print(f"Processed {int(len(self.dataset_context.coco_file.annotations) / i)}% of the shapes")
        coco_json = self.dataset_context.coco_file.model_dump_json()
        with open("annotations_updated.json", "w") as f:
            f.write(coco_json)
        self.dataset_context.dataset_version.import_annotations_coco_file("annotations_updated.json", use_id=True)
