import logging
from typing import Optional, Union, Callable, TypeVar, Dict, Any

from poc.enum import PipelineState, StepState

F = TypeVar("F", bound=Callable[..., None])
logger = logging.getLogger("poc")


class Pipeline:
    ACTIVE_PIPELINE = None

    def __init__(self, name: str, entrypoint: F) -> None:
        self.name = name
        self.entrypoint = entrypoint
        self._state = PipelineState.PENDING
        self._step_statuses: Dict[str, StepState] = {}

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def step_statuses(self) -> Dict[str, StepState]:
        return self._step_statuses

    def __call__(self, *args, **kwargs) -> Any:
        with self:
            return self.entrypoint(*args, **kwargs)

    def __enter__(self):
        if Pipeline.ACTIVE_PIPELINE is not None:
            raise RuntimeError("Another pipeline is already active.")

        Pipeline.ACTIVE_PIPELINE = self
        self._state = PipelineState.RUNNING

    def __exit__(self, exc_type, exc_val, exc_tb):
        Pipeline.ACTIVE_PIPELINE = None

        # Determine final state based on step executions
        if all(status == StepState.SUCCESS for status in self._step_statuses.values()):
            self._state = PipelineState.SUCCESS

        elif any(status == StepState.FAILED for status in self._step_statuses.values()):
            self._state = (
                PipelineState.FAILED
                if self._state != PipelineState.PARTIAL_SUCCESS
                else PipelineState.PARTIAL_SUCCESS
            )

        else:
            self._state = PipelineState.PARTIAL_SUCCESS

    def register_step_execution(self, step_name: str, step_state: StepState) -> None:
        if (
            not step_state == StepState.SUCCESS
            and not self.state == PipelineState.PARTIAL_SUCCESS
        ):
            self._state = PipelineState.FAILED

        elif step_state == StepState.SUCCESS and self.state == PipelineState.FAILED:
            self._state = PipelineState.PARTIAL_SUCCESS

        self._step_statuses[step_name] = step_state


def pipeline(
    _func: Optional["F"] = None,
    name: str | None = None,
) -> Union["Pipeline", Callable[["F"], "Pipeline"]]:
    """Decorator to create a pipeline.

    Args:
        _func: The decorated function.
        name: The name of the pipeline. If left empty, the name of the
            decorated function will be used as a fallback.

    Returns:
        A pipeline instance.
    """

    def inner_decorator(func: "F") -> "Pipeline":
        p = Pipeline(name=name or func.__name__, entrypoint=func)

        p.__doc__ = func.__doc__
        return p

    return inner_decorator if _func is None else inner_decorator(_func)
