import logging
from typing import Union

from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.processing.datalake_collection import DatalakeCollection
from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.model.huggingface.hugging_face_model_context import (
    HuggingFaceModelContext,
)
from src.models.steps.model_prediction.common.CLIP.clip_model_context_predictor import (
    CLIPModelContextPredictor,
)


@step
def clip_datalake_autotagging_processing(
    datalake: Union[DatalakeContext, DatalakeCollection],
    model_context: HuggingFaceModelContext,
):
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_context_predictor = CLIPModelContextPredictor(
        model_context=model_context,
        tags_list=context.processing_parameters.tags_list,
        device=context.processing_parameters.device,
    )
    if isinstance(datalake, DatalakeContext):
        datalake_context = datalake
    elif isinstance(datalake, DatalakeCollection):
        datalake_context = datalake["input"]
    else:
        raise ValueError(
            "Datalake should be either a DatalakeContext or a DatalakeCollection"
        )

    image_inputs, image_paths = model_context_predictor.pre_process_datalake_context(
        datalake_context=datalake_context,
    )
    image_input_batches = model_context_predictor.prepare_batches(
        images=image_inputs,
        batch_size=context.processing_parameters.batch_size,
    )
    image_path_batches = model_context_predictor.prepare_batches(
        images=image_paths,
        batch_size=context.processing_parameters.batch_size,
    )
    batch_results = model_context_predictor.run_inference_on_batches(
        image_batches=image_input_batches
    )
    picsellia_datalake_autotagging_predictions = (
        model_context_predictor.post_process_batches(
            image_batches=image_path_batches,
            batch_results=batch_results,
            datalake_context=datalake_context,
        )
    )
    logging.info(f"Predictions for datalake {datalake_context.datalake.id} done.")

    for (
        picsellia_datalake_autotagging_prediction
    ) in picsellia_datalake_autotagging_predictions:
        if not picsellia_datalake_autotagging_prediction["tag"]:
            continue
        picsellia_datalake_autotagging_prediction["data"].add_tags(
            tags=picsellia_datalake_autotagging_prediction["tag"]
        )

    logging.info(f"Tags added to datalake {datalake_context.datalake.id}.")
