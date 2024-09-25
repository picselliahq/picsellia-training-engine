import os
import re
from difflib import get_close_matches
from typing import List, Dict, Optional

from PIL import Image
from picsellia import Tag, Datalake

from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_prediction.common.model_context_predictor import (
    ModelContextPredictor,
)

import torch
from transformers import AutoProcessor
from transformers import AutoTokenizer


def create_tags(datalake: Datalake, list_tags: list):
    if list_tags:
        for tag_name in list_tags:
            datalake.get_or_create_data_tag(name=tag_name)
    return {k.name: k for k in datalake.list_data_tags()}


def find_label_in_text(picsellia_tags_name: Dict[str, Tag], text: str):
    tags_pattern = "|".join(re.escape(tag) for tag in picsellia_tags_name.keys())
    matches = re.findall(tags_pattern, text)
    return matches


class VLMHuggingFaceModelContextPredictor(ModelContextPredictor[ModelContext]):
    def __init__(
        self,
        model_context: ModelContext,
        model_name: str,
        tags_list: List[str],
    ):
        super().__init__(model_context)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tags_list = tags_list
        self.prompt = self.get_prompt()

    def get_prompt(self) -> str:
        analysis_instruction = "Carefully analyze the image. Based on its content,"

        if self.tags_list:
            base_prompt = f"{analysis_instruction} list all relevant tags that accurately describe the scene, separated by commas. Only include tags that are directly applicable to the image."
        else:
            base_prompt = f"{analysis_instruction} provide the single most relevant tag that best captures the essence of the image."

        options = ", ".join([f'"{tag}"' for tag in self.tags_list])
        prompt = (
            f"{base_prompt} Options to consider are: {options}. Choose appropriately."
        )

        return prompt

    def pre_process_datalake_context(
        self, datalake_context: DatalakeContext, device: str
    ) -> List[str]:
        inputs = []
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
            image = Image.open(image_path)
            input = self.processor(
                images=image, text=self.prompt, return_tensors="pt"
            ).to(device)
            inputs.append(input)
        return inputs

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
            with torch.no_grad():
                outputs = self.model_context.loaded_model.generate(**input)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answers.append(generated_text)
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
            data = datalake_context.datalake.list_data(ids=[data_id])[0]
            closest_label = self.find_most_similar_label(
                llm_answer=prediction, picsellia_tags_name=picsellia_tags_name
            )
            processed_prediction = {"data": data, "tag": closest_label}
            processed_predictions.append(processed_prediction)

        return processed_predictions

    def find_most_similar_label(
        self, llm_answer: str, picsellia_tags_name: Dict[str, Tag]
    ) -> Optional[List[Tag]]:
        closest_labels = []
        llm_tags = find_label_in_text(picsellia_tags_name, llm_answer)
        # llm_tags = [tag.strip() for tag in llm_answer.split(",")]
        for tag in llm_tags:
            close_match = (
                get_close_matches(tag, picsellia_tags_name.keys(), n=1) if tag else []
            )
            if close_match:
                closest_labels.append(picsellia_tags_name[close_match[0]])
        return closest_labels if closest_labels else None
