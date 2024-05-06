from abc import abstractmethod
from enum import Enum, auto
from typing import Any

import numpy as np
from PIL import Image
import torch
import open_clip
from torch._C._te import Tensor
from open_clip.model import CLIP
from torchvision.transforms import transforms
from src import step, Pipeline
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.parameters.processing.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)


class SupportedEmbeddingModels(Enum):
    OPENCLIP = auto()


class EmbeddingModel:
    def __init__(self, device: str):
        self.device = device

    @abstractmethod
    def apply_preprocessing(self, image: Image) -> Any:
        pass

    @abstractmethod
    def encode_image(self, image: Image) -> np.ndarray:
        pass


class OpenClipEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model: CLIP,
        preprocessing: transforms.Compose,
        device: str,
    ):
        super().__init__(device=device)
        self.model = model
        self.preprocessing = preprocessing
        self.device = device

    def apply_preprocessing(self, image: Image) -> Tensor:
        return self.preprocessing(image).unsqueeze(0).to(self.device)

    def encode_image(self, image: Image) -> np.ndarray:
        return self.model.encode_image(image).detach().cpu().numpy()


def is_embedding_model_name_valid(
    source: str, target: SupportedEmbeddingModels
) -> bool:
    """
    Check if the provided embedding model is valid.

    Args:
        source: The provided embedding model.
        target: The target embedding model.

    Returns:
        If the provided embedding model is valid.
    """
    return source.upper() == target.name


@step
def diversified_data_extractor_model_loader() -> EmbeddingModel:
    context: PicselliaProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model_name = context.processing_parameters.embedding_model

    if is_embedding_model_name_valid(
        source=embedding_model_name, target=SupportedEmbeddingModels.OPENCLIP
    ):
        available_model_names = open_clip.list_models()

        model_architecture = context.processing_parameters.model_architecture
        pretrained_weights = context.processing_parameters.pretrained_weights

        if model_architecture not in available_model_names:
            raise ValueError(
                f"The provided model '{context.processing_parameters.model_architecture}' is not available. "
                f"Available models are {available_model_names}."
            )
        elif model_architecture.startswith(open_clip.factory.HF_HUB_PREFIX):
            raise ValueError(
                f"The provided model '{context.processing_parameters.model_architecture}' is a "
                f"HuggingFace model and is not supported yet. "
                f"Please provide a model from the list of available models: {available_model_names}."
            )

        available_pretrained_weights = (
            open_clip.pretrained.list_pretrained_tags_by_model(model=model_architecture)
        )

        if pretrained_weights not in available_pretrained_weights:
            raise ValueError(
                f"The provided pretrained weights '{context.processing_parameters.pretrained_weights}' are not available. "
                f"Available pretrained weights are {open_clip.pretrained.list_pretrained_tags_by_model(model_architecture)}."
            )

        (
            open_clip_model,
            _,
            preprocessing_transformations,
        ) = open_clip.create_model_and_transforms(
            model_name=model_architecture,
            pretrained=pretrained_weights,
        )
        open_clip_model.to(device)
        loaded_model = OpenClipEmbeddingModel(
            model=open_clip_model,
            preprocessing=preprocessing_transformations,
            device=device,
        )

    else:
        raise ValueError(
            f"The provided model '{context.processing_parameters.embedding_model}' is not supported yet. "
            f"Supported models are {[member.name.lower() for member in SupportedEmbeddingModels]}."
        )

    open_clip_model.to(device)
    return loaded_model
