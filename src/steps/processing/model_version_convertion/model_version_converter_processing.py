import os

from src import step
from src.models.model.model_context import ModelContext
from src.models.steps.processing.model_version_convertion.model_version_converter_processing import (
    ModelVersionConversionTargetFramework,
    YoloXModelVersionConverterProcessing,
)


@step
def yolox_model_version_converter_processing(
    model_context: ModelContext,
) -> None:
    processor = YoloXModelVersionConverterProcessing(model_context=model_context)
    processor.process(
        target_frameworks=[ModelVersionConversionTargetFramework.COREML],
        output_path=os.path.join(os.getcwd(), model_context.weights_dir),
    )
