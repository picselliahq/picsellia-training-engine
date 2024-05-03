from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.dataset_context import DatasetContext
from src.models.parameters.processing.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)


@step
def diversified_data_extractor_processing(dataset_context: DatasetContext):
    _: PicselliaProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    # processor = BoundingBoxCropperProcessing(
    #     client=context.client,
    #     input_dataset_context=dataset_context,
    #     label_name_to_extract=context.processing_parameters.label_name_to_extract,
    #     output_dataset_version=context.output_dataset_version,
    #     datalake=context.processing_parameters.datalake,
    #     destination_path=os.path.join(os.getcwd(), str(context.job_id)),
    # )
    # processor.process()
    pass
