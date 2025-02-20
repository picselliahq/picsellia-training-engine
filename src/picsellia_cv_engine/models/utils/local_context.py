import dataclasses
from typing import Dict, Any, Set

from picsellia.types.enums import ProcessingType

from src.picsellia_cv_engine.models.contexts.processing.local_picsellia_processing_context import (
    LocalPicselliaProcessingContext,
)


def infer_type(value: str) -> Any:
    """Infer the type of a value based on its content."""
    try:
        return int(value)  # Check if it's an integer
    except ValueError:
        try:
            return float(value)  # Check if it's a float
        except ValueError:
            return value  # Default to string


def create_local_processing_parameters(processing_parameters: Dict[str, Any]):
    """Create a parameter object with inferred types."""

    @dataclasses.dataclass
    class ProcessingParameters:
        defaulted_keys: Set[str] = dataclasses.field(default_factory=set)

        def __init__(self):
            # Dynamically add attributes and infer types
            for key, value in processing_parameters.items():
                inferred_value = infer_type(value)
                setattr(self, key, inferred_value)

            # Initialize empty defaulted keys for logging purposes
            self.defaulted_keys = set()

        def to_dict(self) -> Dict[str, Any]:
            """Convert the parameters to a dictionary."""
            filtered_dict = {
                key: value
                for key, value in self.__dict__.items()
                if key not in ["defaulted_keys"]
            }
            return dict(sorted(filtered_dict.items()))

        def set_defaulted_key(self, key: str):
            """Log a key as defaulted."""
            self.defaulted_keys.add(key)

    return ProcessingParameters()


def create_local_processing_context(
    api_token: str,
    organization_id: str,
    job_id: str,
    job_type: ProcessingType,
    input_dataset_version_id: str,
    output_dataset_version_name: str,
    processing_parameters: Dict[str, Any],
):
    """
    Create a Picsellia processing context.

    Returns:
        PicselliaProcessingContext: Picsellia processing context object.
    """
    processing_parameters_data = create_local_processing_parameters(
        processing_parameters
    )
    context = LocalPicselliaProcessingContext(
        api_token=api_token,
        organization_id=organization_id,
        job_id=job_id,
        job_type=job_type,
        input_dataset_version_id=input_dataset_version_id,
        output_dataset_version_name=output_dataset_version_name,
        processing_parameters=processing_parameters_data,
    )
    return context
