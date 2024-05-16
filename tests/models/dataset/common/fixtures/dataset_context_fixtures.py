from typing import Callable, Dict, Optional

import pytest
from picsellia import Label
from picsellia.sdk.asset import MultiAsset

from src.models.dataset.common.dataset_context import DatasetContext
from tests.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def mock_dataset_context(
    destination_path: str, mock_dataset_version: Callable
) -> Callable:
    def _mock_dataset_context(
        dataset_metadata: DatasetTestMetadata,
        multi_asset: Optional[MultiAsset] = None,
        labelmap: Optional[Dict[str, Label]] = None,
    ) -> DatasetContext:
        dataset_version = mock_dataset_version(dataset_metadata=dataset_metadata)
        dataset_context = DatasetContext(
            dataset_name=dataset_metadata.attached_name,
            dataset_version=dataset_version,
            destination_path=destination_path,
            multi_asset=multi_asset,
            labelmap=labelmap,
        )
        return dataset_context

    return _mock_dataset_context
