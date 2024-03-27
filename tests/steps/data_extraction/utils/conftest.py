import dataclasses
from typing import Optional

from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName

pytest_plugins = [
    "tests.steps.fixtures.initialize_integration_tests_fixtures",
    "tests.steps.data_extraction.fixtures.classification_dataset_fixtures",
    "tests.steps.data_extraction.fixtures.dataset_collection_fixtures",
    "tests.steps.data_extraction.fixtures.dataset_handler_fixtures",
]


@dataclasses.dataclass
class DatasetTestMetadata:
    def __init__(
        self,
        dataset_split_name: DatasetSplitName,
        dataset_type: InferenceType,
        attached_name: Optional[str] = None,
    ):
        self.dataset_split_name = dataset_split_name
        self.dataset_type = dataset_type
        self.attached_name = attached_name or dataset_split_name.value
