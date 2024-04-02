from src.models.parameters.parameters import Parameters
import pytest


class ConcreteParameters(Parameters):
    """A concrete implementation of the abstract Parameters class for testing."""

    pass


@pytest.fixture
def log_data():
    return {
        "key1": "value1",
        "key2": 100,
        "key3": True,
        "key4": {"nested_key1": 63, "nested_key2": 42.0},
    }


@pytest.fixture
def parameters(log_data):
    return ConcreteParameters(log_data)
