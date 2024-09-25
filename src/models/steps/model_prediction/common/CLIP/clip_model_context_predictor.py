import os
from typing import List, Dict, Optional, Tuple
from uuid import UUID

from PIL import Image
from picsellia import Tag, Datalake

from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_prediction.common.model_context_predictor import (
    ModelContextPredictor,
)

import torch
from transformers import CLIPProcessor

def create_tags(datalake: Datalake, list_tags: list):
    if list_tags:
        for tag_name in list_tags:
            datalake.get_or_create_data_tag(name=tag_name)
    return {k.name: k for k in datalake.list_data_tags()}


class VLMHuggingFaceModelContextPredictor(ModelContextPredictor[ModelContext]):
    def __init__(
        self,
        model_context: ModelContext,
        tags_list: List[str],
    ):
        super().__init__(model_context)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tags_list = tags_list

    def pre_process_datalake_context(
        self, datalake_context: DatalakeContext, device: str
    ) -> Tuple[List, List[str]]:
        inputs = []
        image_paths = []
        if device.startswith("cuda") and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        elif device.startswith("cuda") and not torch.cuda.is_available():
            print("Using CPU")
            device = torch.device("cpu")
        else:
            print("Using CPU")
            device = torch.device("cpu")
        for image_name in os.listdir(datalake_context.image_dir):
            image_path = os.path.join(datalake_context.image_dir, image_name)
            image_paths.append(image_path)
            image = Image.open(image_path)
            input = self.processor(images=image, text=[tag.replace("_", " ") for tag in self.tags_list], return_tensors="pt", padding=True).to(device)
            inputs.append(input)
        return inputs, image_paths

    def prepare_batches(self, image_inputs: List, batch_size: int) -> List[List[str]]:
        """
        Divides the list of image paths into smaller batches of a specified size.

        Args:
            image_paths (List[str]): A list of image file paths to be split into batches.
            batch_size (int): The size of each batch.

        Returns:
            List[List[str]]: A list of batches, each containing a list of image file paths.
        """
        return [
            image_inputs[i : i + batch_size]
            for i in range(0, len(image_inputs), batch_size)
        ]

    def run_inference_on_batches(
        self, image_batches: List[List[str]]
    ) -> List[List[str]]:
        all_batch_results = []

        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_inputs=batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_inputs: List) -> List[str]:
        answers = []
        for input in batch_inputs:
            outputs = self.model_context.loaded_model(**input)
            probs = outputs.logits_per_image.softmax(dim=1)
            predicted_label = self.tags_list[probs.argmax().item()]
            print(f'predicted_label: {predicted_label}')
            answers.append(predicted_label)
        return answers

    def post_process_batches(
        self,
        image_batches: List[List[str]],
        batch_results: List[List[str]],
        datalake_context: DatalakeContext,
    ) -> List[Dict]:
        all_predictions = []

        picsellia_tags_name = create_tags(
            datalake=datalake_context.datalake, list_tags=self.tags_list
        )

        for batch_result, batch_paths in zip(batch_results, image_batches):
            all_predictions.extend(
                self._post_process(
                    image_paths=batch_paths,
                    batch_prediction=batch_result,
                    datalake_context=datalake_context,
                    picsellia_tags_name=picsellia_tags_name,
                )
            )
        return all_predictions

    def _post_process(
        self,
        image_paths: List[str],
        batch_prediction: List[str],
        datalake_context: DatalakeContext,
        picsellia_tags_name: Dict[str, Tag],
    ) -> List[Dict]:
        processed_predictions = []

        for image_path, prediction in zip(image_paths, batch_prediction):
            data_id = os.path.basename(image_path).split(".")[0]
            data = datalake_context.datalake.list_data(ids=[UUID(data_id)])[0]
            picsellia_tag = self.get_picsellia_tag(
                prediction=prediction, picsellia_tags_name=picsellia_tags_name
            )
            processed_prediction = {"data": data, "tag": picsellia_tag}
            processed_predictions.append(processed_prediction)

        return processed_predictions

    def get_picsellia_tag(
        self, prediction: str, picsellia_tags_name: Dict[str, Tag]
    ) -> Optional[List[Tag]]:
        return picsellia_tags_name[prediction]
