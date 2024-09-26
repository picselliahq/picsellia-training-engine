import os
from typing import List, Dict, Tuple, Any
from uuid import UUID

from PIL import Image
from picsellia import Tag, Datalake

from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.model.huggingface.hugging_face_model_context import (
    HuggingFaceModelContext,
)
from src.models.steps.model_loading.common.CLIP.clip_model_context_loader import (
    get_device,
)
from src.models.steps.model_prediction.common.model_context_predictor import (
    ModelContextPredictor,
)


def create_tags(datalake: Datalake, list_tags: list):
    """
    Creates or retrieves tags from the Datalake.

    Args:
        datalake (Datalake): The datalake object to interact with.
        list_tags (list): List of tags to create or retrieve.

    Returns:
        dict: A dictionary of tag names and Tag objects.
    """
    if list_tags:
        for tag_name in list_tags:
            datalake.get_or_create_data_tag(name=tag_name)
    return {k.name: k for k in datalake.list_data_tags()}


class CLIPModelContextPredictor(ModelContextPredictor[HuggingFaceModelContext]):
    """
    A class to handle the prediction process for CLIP model within a given model context.

    Args:
        model_context (HuggingFaceModelContext): The model context containing the HuggingFace model and processor.
        tags_list (List[str]): A list of tags used for image classification.
        device (str): The device ('cpu' or 'gpu') on which to run the model.
    """

    def __init__(
        self,
        model_context: HuggingFaceModelContext,
        tags_list: List[str],
        device: str = "cuda:0",
    ):
        """
        Initializes the CLIPModelContextPredictor.

        Args:
            model_context (HuggingFaceModelContext): The context of the model to be used.
            tags_list (List[str]): List of tags for inference.
            device (str): The device ('cpu' or 'gpu') on which to run the model.
        """
        super().__init__(model_context)
        if not hasattr(self.model_context, "loaded_processor"):
            raise ValueError("The model context does not have a processor attribute.")
        self.tags_list = tags_list
        self.device = get_device(device)

    def pre_process_datalake_context(
        self, datalake_context: DatalakeContext
    ) -> Tuple[List, List[str]]:
        """
        Pre-processes images from the datalake context by converting them into inputs for the model.

        Args:
            datalake_context (DatalakeContext): The context containing the directory of images.
            device (str): The device ('cpu' or 'gpu') on which to run the model.

        Returns:
            Tuple[List, List[str]]: A tuple containing the list of preprocessed inputs and image paths.
        """
        inputs = []
        image_paths = []
        for image_name in os.listdir(datalake_context.image_dir):
            image_path = os.path.join(datalake_context.image_dir, image_name)
            image_paths.append(image_path)
            image = Image.open(image_path)

            input = self.model_context.loaded_processor(
                images=image,
                text=[tag.replace("_", " ") for tag in self.tags_list],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            inputs.append(input)

        return inputs, image_paths

    def prepare_batches(self, images: List[Any], batch_size: int) -> List[List[str]]:
        """
        Splits the given images into batches of specified size.

        Args:
            images (List[Any]): A list of images to split into batches.
            batch_size (int): The size of each batch.

        Returns:
            List[List[str]]: A list of image batches.
        """
        return [images[i : i + batch_size] for i in range(0, len(images), batch_size)]

    def run_inference_on_batches(
        self, image_batches: List[List[str]]
    ) -> List[List[str]]:
        """
        Runs inference on each batch of images.

        Args:
            image_batches (List[List[str]]): List of image batches for inference.

        Returns:
            List[List[str]]: A list of predicted labels for each batch.
        """
        all_batch_results = []

        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_inputs=batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_inputs: List) -> List[str]:
        """
        Runs the model inference on a batch of inputs.

        Args:
            batch_inputs (List): A batch of pre-processed image inputs.

        Returns:
            List[str]: A list of predicted labels for the batch.
        """
        answers = []
        for input in batch_inputs:
            outputs = self.model_context.loaded_model(**input)
            probs = outputs.logits_per_image.softmax(dim=1)
            predicted_label = self.tags_list[probs.argmax().item()]
            answers.append(predicted_label)
        return answers

    def post_process_batches(
        self,
        image_batches: List[List[str]],
        batch_results: List[List[str]],
        datalake_context: DatalakeContext,
    ) -> List[Dict]:
        """
        Post-processes the batch predictions by mapping them to Picsellia tags and generating a final output.

        Args:
            image_batches (List[List[str]]): List of image batches.
            batch_results (List[List[str]]): List of batch prediction results.
            datalake_context (DatalakeContext): The datalake context for processing.

        Returns:
            List[Dict]: A list of dictionaries containing processed predictions.
        """
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
        """
        Maps the predictions to Picsellia tags and returns processed predictions.

        Args:
            image_paths (List[str]): List of image paths.
            batch_prediction (List[str]): List of predictions for each image.
            datalake_context (DatalakeContext): The datalake context for retrieving data.
            picsellia_tags_name (Dict[str, Tag]): A dictionary of Picsellia tags.

        Returns:
            List[Dict]: A list of dictionaries containing data and their corresponding Picsellia tags.
        """
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
    ) -> Tag:
        """
        Retrieves the Picsellia tag corresponding to the prediction.

        Args:
            prediction (str): The predicted tag name.
            picsellia_tags_name (Dict[str, Tag]): A dictionary mapping tag names to Tag objects.

        Returns:
            Tag: The corresponding Picsellia Tag object.

        Raises:
            ValueError: If the predicted tag is not found in Picsellia tags.
        """
        if prediction not in picsellia_tags_name:
            raise ValueError(f"Tag {prediction} not found in Picsellia tags.")
        return picsellia_tags_name[prediction]
