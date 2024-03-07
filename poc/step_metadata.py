from uuid import UUID

from poc.enum import StepState


class StepMetadata:
    def __init__(
        self,
        id: UUID,
        name: str,
        state: StepState,
        execution_time: float,
    ) -> None:
        self.id = id
        self.name = name
        self.state = state
        self.execution_time = execution_time

    def __repr__(self):
        return f"StepMetadata('{self.id}', '{self.name}', {self.state}, {self.execution_time})"
