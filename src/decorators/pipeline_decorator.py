import ast
import inspect
from typing import Optional, Union, Callable, TypeVar, Any, List, Tuple

from tabulate import tabulate

from src.enums import PipelineState, StepState
from src.logger import LoggerManager
from src.models.steps.step_metadata import StepMetadata

F = TypeVar("F", bound=Callable[..., None])


class Pipeline:
    ACTIVE_PIPELINE: Optional["Pipeline"] = None
    STEPS_REGISTRY: dict[str, StepMetadata] = {}

    def __init__(
        self,
        context: Any,
        name: str,
        log_folder_path: Optional[str],
        remove_logs_on_completion: bool,
        entrypoint: F,
    ) -> None:
        """Initializes a pipeline.

        Args:
            context: The context of the pipeline. This object can be used to store and share data between steps.
                It will be accessible by calling the method `Pipeline.get_active_context()`.
            name: The name of the pipeline.
            log_folder_path: The path to the log folder. If left empty, a temporary folder will be created.
            remove_logs_on_completion: Whether to remove the logs on completion. Defaults to True.
            entrypoint: The entrypoint of the pipeline. This is the function that will be called when the pipeline is run.
        """
        self._context = context
        self.name = name
        self.logger_manager = LoggerManager(
            pipeline_name=name, log_folder_root_path=log_folder_path
        )
        self.remove_logs_on_completion = remove_logs_on_completion
        self.entrypoint = entrypoint

        self._is_pipeline_initialized = False
        self._state = PipelineState.PENDING
        self._registered_steps_metadata: [StepMetadata] = []

        self.initialization_log_file_path = None

    def __call__(self, *args, **kwargs) -> Any:
        with self:
            self._analyze_and_register_steps()
            self.finalize_initialization()
            self.log_pipeline_context()

            return self.entrypoint(*args, **kwargs)

    def __enter__(self):
        if Pipeline.ACTIVE_PIPELINE is not None:
            raise RuntimeError(
                "Another pipeline is already active."
                "A pipeline cannot be run within another pipeline."
            )

        Pipeline.ACTIVE_PIPELINE = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Pipeline.ACTIVE_PIPELINE = None

        if self.remove_logs_on_completion:
            self.logger_manager.clean()

    @property
    def is_initialized(self) -> bool:
        return self._is_pipeline_initialized

    @property
    def state(self) -> PipelineState:
        """
        The state of the pipeline is determined by the state of its steps.
        """
        if all(
            step_metadata.state == StepState.PENDING
            for step_metadata in self.steps_metadata
        ):
            return PipelineState.PENDING

        elif all(
            step_metadata.state == StepState.SUCCESS
            for step_metadata in self.steps_metadata
        ):
            return PipelineState.SUCCESS

        elif all(
            step_metadata.state
            in [StepState.RUNNING, StepState.PENDING, StepState.SUCCESS]
            for step_metadata in self.steps_metadata
        ):
            return PipelineState.RUNNING

        else:
            for step_metadata in self.steps_metadata:
                if step_metadata.state in [StepState.FAILED, StepState.SKIPPED]:
                    if any(
                        s.state == StepState.SUCCESS
                        for s in self.steps_metadata[
                            self.steps_metadata.index(step_metadata) + 1 :
                        ]
                    ):
                        return PipelineState.PARTIAL_SUCCESS
                    return PipelineState.FAILED

    @property
    def steps_metadata(self) -> List[StepMetadata]:
        return self._registered_steps_metadata

    def finalize_initialization(self) -> None:
        self.logger_manager.configure(steps_metadata=self.steps_metadata)
        self._is_pipeline_initialized = True
        self._state = PipelineState.RUNNING

        self.initialization_log_file_path = (
            self.logger_manager.configure_pipeline_initialization_log_file()
        )
        self.logger_manager.prepare_logger(
            log_file_path=self.initialization_log_file_path
        )

    def log_pipeline_context(self):
        self.log_pipeline_info(
            f"Pipeline \033[94m{self.name}\033[0m is starting with the following context:"
        )

        context_dict = self._parse_context_to_dict(self._context)

        # Separate flat parameters from nested ones
        flat_parameters, nested_parameters = self._extract_parameters_from_context_dict(
            context_dict
        )

        # Log flat parameters
        if flat_parameters:
            markdown_table = self._compute_markdown_table("parameters", flat_parameters)
            self.log_pipeline_info(f"{markdown_table}\n")

        # Log each nested dictionary under its key
        for key, nested_dict in nested_parameters.items():
            markdown_table = self._compute_markdown_table(key, nested_dict)
            self.log_pipeline_info(f"{markdown_table}\n")

    def log_pipeline_info(self, log_content: str) -> None:
        self.logger_manager.logger.info(f"{log_content}")

    def register_active_step_metadata(self, step_metadata: StepMetadata) -> None:
        self._registered_steps_metadata.append(step_metadata)

    def _analyze_and_register_steps(self) -> None:
        """Analyze the pipeline entrypoint function to identify and register step calls."""
        try:
            src = inspect.getsource(self.entrypoint)
        except TypeError as e:
            raise ValueError(
                f"The provided entrypoint cannot be scanned: {str(e)}"
            ) from e

        tree = ast.parse(src)

        pipeline_instance = self

        class StepCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    step_func_name = node.func.id
                    if step_func_name in Pipeline.STEPS_REGISTRY:
                        metadata = Pipeline.STEPS_REGISTRY[step_func_name]
                        pipeline_instance.register_active_step_metadata(metadata)

                self.generic_visit(node)

        visitor = StepCallVisitor()
        visitor.visit(tree)

    def _compute_markdown_table(self, category: str, context: dict) -> str:
        headers = (
            [category, "values"]
            if isinstance(context, dict)
            and not any(isinstance(val, dict) for val in context.values())
            else ["parameters", "values"]
        )
        data = [
            [key, (value if value is not None else "None")]
            for key, value in context.items()
        ]
        return tabulate(
            tabular_data=data,
            headers=headers,
            tablefmt="github",
        )

    def _extract_parameters_from_context_dict(self, context: dict) -> Tuple[dict, dict]:
        flat_parameters = {}
        nested_parameters = {}

        for key, value in context.items():
            if isinstance(value, dict):
                nested_parameters[key] = value
            else:
                flat_parameters[key] = value

        return flat_parameters, nested_parameters

    def _parse_context_to_dict(self, context: Any) -> Optional[dict]:
        if hasattr(context, "to_dict") and callable(getattr(context, "to_dict")):
            return context.to_dict()
        elif isinstance(context, dict):
            return context
        else:
            return None

    @staticmethod
    def get_active_context() -> Any:
        if Pipeline.ACTIVE_PIPELINE is None:
            raise RuntimeError(
                "No current pipeline running."
                "A step must be run within a function decorated with @pipeline."
            )

        if Pipeline.ACTIVE_PIPELINE._context is None:
            raise RuntimeError(
                "No context has been set for the current pipeline."
                "A context must be set when creating a pipeline by passing it as a parameter to the decorator."
                "Example: '@pipeline(context=MyContext())'"
            )

        return Pipeline.ACTIVE_PIPELINE._context

    @staticmethod
    def register_step_metadata(step_metadata: StepMetadata) -> None:
        step_name = step_metadata.name

        if step_name in Pipeline.STEPS_REGISTRY:
            raise ValueError(
                f"More than one step is called '{step_name}'. The step names must be unique."
            )

        Pipeline.STEPS_REGISTRY[step_name] = step_metadata


def pipeline(
    _func: Optional["F"] = None,
    context: Optional[Any] = None,
    name: Optional[str] = None,
    log_folder_path: Optional[str] = None,
    remove_logs_on_completion: bool = True,
) -> Union["Pipeline", Callable[["F"], "Pipeline"]]:
    """Decorator to create a pipeline.

    Args:
        _func: The decorated function.
        context: The context of the pipeline. This object can be used to store and share data between steps.
            It will be accessible by calling the method `Pipeline.get_active_context()`.
        name: The name of the pipeline. If left empty, the name of the
            decorated function will be used as a fallback.
        log_folder_path: The path to the log folder. If left empty, a temporary folder will be created.
        remove_logs_on_completion: Whether to remove the logs on completion. Defaults to True.

    Returns:
        A pipeline instance.
    """

    def inner_decorator(func: "F") -> "Pipeline":
        p = Pipeline(
            context=context,
            name=name or func.__name__,
            log_folder_path=log_folder_path,
            remove_logs_on_completion=remove_logs_on_completion,
            entrypoint=func,
        )

        p.__doc__ = func.__doc__
        return p

    return inner_decorator if _func is None else inner_decorator(_func)
