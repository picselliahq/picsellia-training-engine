# type: ignore

from src import pipeline
from src.models.contexts.processing.picsellia_datalake_processing_context import (
    PicselliaDatalakeProcessingContext,
)
from src.models.parameters.processing.processing_datalake_autotagging_parameters import (
    ProcessingDatalakeAutotaggingParameters,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_datalake_extractor,
)
from src.steps.model_loading.common.CLIP.clip_model_context_loader import (
    clip_model_context_loader,
)
from src.steps.processing.autotagging.clip_datalake_autotagging import (
    clip_datalake_autotagging_processing,
)
from src.steps.weights_extraction.common.hugging_face_weights_extractor import (
    hugging_face_model_context_extractor,
)


def get_context() -> (
    PicselliaDatalakeProcessingContext[ProcessingDatalakeAutotaggingParameters]
):
    return PicselliaDatalakeProcessingContext(
        processing_parameters_cls=ProcessingDatalakeAutotaggingParameters
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def datalake_autotagging_processing_pipeline() -> None:
    datalake = processing_datalake_extractor()
    model_context = hugging_face_model_context_extractor(
        hugging_face_model_name="openai/clip-vit-base-patch32"
    )
    model_context = clip_model_context_loader(model_context=model_context)
    clip_datalake_autotagging_processing(datalake=datalake, model_context=model_context)


if __name__ == "__main__":
    import torch
    import os

    torch.set_num_threads(os.cpu_count() - 1)
    datalake_autotagging_processing_pipeline()
