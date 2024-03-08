from typing import Union
from uuid import UUID

from poc.enum import StepState


class StepMetadata:
    def __init__(
        self,
        id: UUID,
        name: str,
        state: StepState,
        execution_time: float,
        log_file_path: Union[str, None] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.state = state
        self.execution_time = execution_time
        self.log_file_path = log_file_path

    def __repr__(self):
        return f"StepMetadata('{self.id}', '{self.name}', {self.state}, {self.execution_time}, {self.log_file_path})"
