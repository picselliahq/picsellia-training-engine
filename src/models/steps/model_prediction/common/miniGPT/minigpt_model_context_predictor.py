# type: ignore
import os
import re
from difflib import get_close_matches
from typing import List, Dict, Optional

import yaml
from PIL import Image
from picsellia import Tag, Datalake

from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_prediction.common.model_context_predictor import (
    ModelContextPredictor,
)
from src.pipelines.datalake_autotagging.MiniGPT4.minigpt4.conversation.conversation import (
    Conversation,
    SeparatorStyle,
)


def resize_image(image_path: str, size: int):
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((size, size))
    return resized_image


def get_conversation(architecture: str):
    if architecture == "minigpt4":
        CONV_VISION = Conversation(
            system="Give the following image: <Img>ImageContent</Img>. "
            "You will be able to see the image once I provide it to you. Please answer my questions.",
            roles=["Human: ", "Assistant: "],
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
    elif architecture == "minigpt4_llama2":
        CONV_VISION = Conversation(
            system="Give the following image: <Img>ImageContent</Img>. "
            "You will be able to see the image once I provide it to you. Please answer my questions.",
            roles=["<s>[INST] ", " [/INST] "],
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
    elif architecture == "minigpt_v2":
        CONV_VISION = Conversation(
            system="",
            roles=["<s>[INST] ", " [/INST]"],
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
    else:
        raise ValueError("Model version not supported")
    return CONV_VISION


def create_tags(datalake: Datalake, list_tags: list):
    if list_tags:
        for tag_name in list_tags:
            datalake.get_or_create_data_tag(name=tag_name)
    return {k.name: k for k in datalake.list_data_tags()}


def find_label_in_text(picsellia_tags_name: Dict[str, Tag], text: str):
    tags_pattern = "|".join(re.escape(tag) for tag in picsellia_tags_name.keys())
    matches = re.findall(tags_pattern, text)
    return matches


class MiniGPTModelContextPredictor(ModelContextPredictor[ModelContext]):
    def __init__(
        self,
        model_context: ModelContext,
        tags_list: List[str],
        temperature: float = 0.2,
        max_new_tokens: int = 500,
        max_length: int = 2000,
    ):
        super().__init__(model_context)
        self.config_path = model_context.config_path
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.model_architecture = self.config["architecture"]
        self.image_size = self.config["image_size"]
        self.conversation = get_conversation(architecture=self.model_architecture)
        self.tags_list = tags_list
        self.prompt = self.get_prompt()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

    def get_prompt(self) -> str:
        analysis_instruction = "Carefully analyze the image. Based on its content,"

        if self.tags_list:
            base_prompt = f"{analysis_instruction} list all relevant tags that accurately describe the scene, separated by commas. Only include tags that are directly applicable to the image."
        else:
            base_prompt = f"{analysis_instruction} provide the single most relevant tag that best captures the essence of the image."

        if self.model_architecture == "minigpt4":
            options = ", ".join([f'"{tag}"' for tag in self.tags_list])
            prompt = f"[vqa] {base_prompt} Consider the following tags as options: {options}. Select wisely."
        elif self.model_architecture == "minigpt_v2":
            options = ", ".join([f'"{tag}"' for tag in self.tags_list])
            prompt = f"[vqa] {base_prompt} Options to consider are: {options}. Choose appropriately."
        else:
            raise (ValueError("Model version not supported"))

        return prompt

    def pre_process_datalake_context(
        self, datalake_context: DatalakeContext
    ) -> List[str]:
        image_paths = []
        for image_name in os.listdir(datalake_context.image_dir):
            image_path = os.path.join(datalake_context.image_dir, image_name)
            image_paths.append(image_path)
        return image_paths

    def prepare_batches(
        self, image_paths: List[str], batch_size: int
    ) -> List[List[str]]:
        """
        Divides the list of image paths into smaller batches of a specified size.

        Args:
            image_paths (List[str]): A list of image file paths to be split into batches.
            batch_size (int): The size of each batch.

        Returns:
            List[List[str]]: A list of batches, each containing a list of image file paths.
        """
        return [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

    def run_inference_on_batches(
        self, image_batches: List[List[str]]
    ) -> List[List[str]]:
        all_batch_results = []

        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_paths: List[str]) -> List[str]:
        llm_messages = []
        for image_path in batch_paths:
            chat_state = self.conversation.copy()
            img_list = self.prepare_img_list(image_path, chat_state)
            self.model_context.loaded_model.ask(self.prompt, chat_state)
            llm_message = self.model_context.loaded_model.answer(
                conv=chat_state,
                img_list=img_list,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_length=self.max_length,
            )[0]
            llm_messages.append(llm_message)
        return llm_messages

    def prepare_img_list(self, image_path: str, chat_state: Conversation):
        img_list = []
        resized_image = resize_image(image_path=image_path, size=self.image_size)
        _ = self.model_context.loaded_model.chat.upload_img(
            image=resized_image, conv=chat_state, img_list=img_list
        )
        self.model_context.loaded_model.chat.encode_img(image=img_list)
        return img_list

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
            processed_prediction = {
                "data": data,
                "tag": closest_label,
                "confidence": 1.0,
            }
            processed_predictions.append(processed_prediction)

        return processed_predictions

    def find_most_similar_label(
        self, llm_answer: str, picsellia_tags_name: Dict[str, Tag]
    ) -> Optional[str]:
        closest_labels = []
        if self.model_architecture == "minigpt4":
            llm_tags = find_label_in_text(picsellia_tags_name, llm_answer)
        elif self.model_architecture == "minigpt_v2":
            llm_tags = [tag.strip() for tag in llm_answer.split(",")]
        else:
            llm_tags = []
        for tag in llm_tags:
            close_match = (
                get_close_matches(tag, picsellia_tags_name.keys(), n=1) if tag else []
            )
            if close_match:
                closest_labels.append(picsellia_tags_name[close_match[0]])
        return closest_labels if closest_labels else None
