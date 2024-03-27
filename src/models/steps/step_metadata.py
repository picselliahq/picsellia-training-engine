from typing import Union
from uuid import UUID

from src.enums import StepState


class StepMetadata:
    def __init__(
        self,
        id: UUID,
        name: str,
        display_name: str,
        state: StepState,
        log_file_path: Union[str, None] = None,
    ) -> None:
        self.id = id
        self.name = name
        self.display_name = display_name
        self.state = state
        self.execution_time = 0.0
        self.log_file_path = log_file_path
        self.index = None

    def __repr__(self):
        return (
            f"StepMetadata("
            f"'{self.id}', '{self.name}', '{self.display_name}', {self.state},"
            f"{self.execution_time}, {self.log_file_path}, {self.index}"
            f")"
        )
