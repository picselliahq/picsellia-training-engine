from typing import TypeVar


class ModelCollection:
    pass


TModelCollection = TypeVar("TModelCollection", bound=ModelCollection)
