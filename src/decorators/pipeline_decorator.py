import ast
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, overload

from tabulate import tabulate  # type: ignore

from src import Colors
from src.enums import PipelineState, StepState
from src.logger import LoggerManager
from src.models.steps.step_metadata import StepMetadata

F = TypeVar("F", bound=Callable[..., Any])


class Pipeline:
    ACTIVE_PIPELINE: Optional["Pipeline"] = None
    STEPS_REGISTRY: Dict[str, StepMetadata] = {}

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
            entrypoint: The entrypoint of the pipeline.
                This is the function that will be called when the pipeline is run.
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
        self._registered_steps_metadata: List[StepMetadata] = []

        self.initialization_log_file_path: Optional[str] = None

    def __call__(self, *args, **kwargs) -> Any:
        """Handles the pipeline call.

        This method first analyses and registers the steps of the pipeline.
        Then, it configures the logging, flags the pipeline as running,
        logs the pipeline context, and runs the entrypoint.

        Args:
            *args: Arguments to be passed to the entrypoint function.
            **kwargs: Keyword arguments to be passed to the entrypoint function.

        Returns:
            The outputs of the entrypoint function call.
        """
        with self:
            self._scan_steps()
            self._configure_logging()
            self._flag_pipeline(state=PipelineState.RUNNING)
            self._log_pipeline_context()

            return self.entrypoint(*args, **kwargs)

    def __enter__(self):
        """Activate the pipeline context.

        Raises:
            RuntimeError: If another pipeline is already active.
                Typically, occurs when a pipeline is run within another pipeline.
        """
        if Pipeline.ACTIVE_PIPELINE is not None:
            raise RuntimeError(
                "Another pipeline is already active."
                "A pipeline cannot be run within another pipeline."
            )

        Pipeline.ACTIVE_PIPELINE = self

    def __exit__(self, *args: Any):
        """Deactivates the pipeline context and clean the log folder is requested.

        Args:
            *args: The arguments passed to the context exit handler.
        """
        Pipeline.ACTIVE_PIPELINE = None

        if self.remove_logs_on_completion:
            self.logger_manager.clean()

    @property
    def state(self) -> PipelineState:
        """The state of the pipeline.

        The pipeline's state is determined by the states of its steps.
        The pipeline can be in one of the following states:

        - PENDING: All the steps are pending.
        - RUNNING: At least one step is running, and no step has failed.
        - SUCCESS: All the steps have succeeded.
        - FAILED: At least one step has failed and no step has succeeded after it. By default, a step is skipped if a
            previous step has failed. This behavior can be changed by setting the `continue_on_failure` parameter to
            `True` when defining a step.
        - PARTIAL_SUCCESS: At least one step has succeeded after a failed step.

        Returns:
            The state of the pipeline.
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

        raise RuntimeError(
            "Pipeline state could not be determined."
        )  # pragma: no cover

    @property
    def steps_metadata(self) -> List[StepMetadata]:
        """All the pipeline's steps' metadata.

        Returns:
            All the pipeline's steps' metadata.
        """
        return self._registered_steps_metadata

    def log_pipeline_info(self, log_content: str) -> None:
        """Log the provided content inside the pipeline log file.

        Args:
            log_content: The content to log.
        """
        self.logger_manager.logger.info(f"{log_content}")

    def log_pipeline_warning(self, log_content: str) -> None:
        """Log the provided content inside the pipeline log file.

        Args:
            log_content: The content to log.
        """
        self.logger_manager.logger.warning(f"{log_content}")

    def register_active_step_metadata(self, step_metadata: StepMetadata) -> None:
        """Register the metadata of a step found during the pipeline scan.

        Args:
            step_metadata: The metadata of the step to register.
        """
        self._registered_steps_metadata.append(step_metadata)

    def _compute_markdown_table(self, category: str, context: dict) -> str:
        """Format a dictionary as a Markdown table with two columns: keys and values.

        Args:
            category: The category of the provided context. This will be the header of the first column.
                For example, "hyperparameters", "augmentation_parameters, etc.
            context: The context to format.

        Returns:
            The context formatted as a Markdown table.
        """
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

    def _configure_logging(self) -> None:
        """Configures the logging for the pipeline.

        This method configures the pipeline logger, the pipeline's dedicated log file and prepares the logger if
        the pipeline needs to log something before the first step is run.
        """
        self.logger_manager.configure_log_files(steps_metadata=self.steps_metadata)

        self.initialization_log_file_path = (
            self.logger_manager.configure_pipeline_initialization_log_file()
        )
        self.logger_manager.prepare_logger(
            log_file_path=self.initialization_log_file_path
        )

    def _extract_parameters_from_context_dict(
        self, context: Dict[Any, Any]
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        """Extracts flat parameters and nested parameters from a context dictionary.

        For example, given the following context: `{'nested': {'learning_rate': 0.01}, 'flat': "value"}`,
        the output will be :

        - flat_parameters: `{'flat': "value"}`
        - nested_parameters: `{'nested': {'learning_rate': 0.01}}`

        Args:
            context: The context dictionary to extract the parameters from.

        Returns:
            A tuple containing the flat parameters and the nested parameters.
        """
        flat_parameters: Dict[Any, Any] = {}
        nested_parameters: Dict[Any, Any] = {}

        for key, value in context.items():
            if isinstance(value, dict):
                nested_parameters[key] = value
            else:
                flat_parameters[key] = value

        return flat_parameters, nested_parameters

    def _flag_pipeline(self, state: PipelineState) -> None:
        """Flags the pipeline with the provided state.

        Args:
            state: The state to flag the pipeline with.
        """
        self._state = state

    def _log_pipeline_context(self):
        """Log the pipeline context.

        This method logs an introduction sentence, followed by the pipeline's context.

        - If the context exposes a `to_dict` method, then it will be called to convert the context to a dictionary.
        - If the context is already a dictionary, it will be used as is.
        - If the context is neither a dictionary nor exposes a `to_dict` method, then it will be ignored.

        Typically, a context would look like this: `{'hyperparameters': {'learning_rate': 0.01, 'batch_size': 32}}`.
        In this case, the context will be printed as a Markdown table with the following format:

        | hyperparameters   | values   |
        |-------------------|----------|
        | learning_rate     | 0.01     |
        | batch_size        | 32       |
        """
        self.log_pipeline_info(
            log_content=f"Pipeline {Colors.BLUE}{self.name}{Colors.ENDC} is starting with the following context:"
        )

        context_dict = self._parse_context_to_dict(self._context)

        if context_dict is None:
            self.log_pipeline_warning(
                log_content="The provided context does not expose a `to_dict` method, "
                "therefore it will could not be logged. "
                "Try to implement a `to_dict` method in your context or provide a dictionary."
            )
            return

        flat_parameters, nested_parameters = self._extract_parameters_from_context_dict(
            context_dict
        )

        # Log flat parameters
        if flat_parameters:
            markdown_table = self._compute_markdown_table("parameters", flat_parameters)
            self.log_pipeline_info(log_content=f"{markdown_table}\n")

        # Log each nested dictionary under its key
        for key, nested_dict in nested_parameters.items():
            markdown_table = self._compute_markdown_table(key, nested_dict)
            self.log_pipeline_info(log_content=f"{markdown_table}\n")

    def _parse_context_to_dict(self, context: Any) -> Optional[Dict[Any, Any]]:
        """Parse the context to a dictionary.

        This method only works if the context exposes a `to_dict` method or is already a dictionary.

        Args:
            context: The context to parse.

        Returns:
            The context as a dictionary.
        """
        if hasattr(context, "to_dict") and callable(getattr(context, "to_dict")):
            return context.to_dict()
        elif isinstance(context, dict):
            return context
        else:
            return None

    def _scan_steps(self) -> None:
        """Analyze the pipeline entrypoint function to identify and register step calls.

        The pipeline is scanned using the `inspect` module to extract the source code of the entrypoint function.
        Each note is matched with the global STEPS_REGISTRY to identify the steps that are called.

        Raises:
            ValueError: If the provided entrypoint cannot be scanned.
        """
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

    @staticmethod
    def get_active_context() -> Any:
        """Get the context of the currently running pipeline.

        Returns:
            The context of the currently running pipeline.

        Raises:
            RuntimeError: If no current pipeline is running.
            RuntimeError: If no context has been set for the current pipeline.
        """
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
        """Register a step metadata into the global steps' registry.

        This method is only used to register the steps' metadata within the global steps registry,
        typically when a step is defined. This registry will then be used during the pipeline scan to identify the steps
        used inside.

        Args:
            step_metadata: The metadata of the step to register.

        Raises:
            ValueError: If a step with the same name has already been registered. The step names must be unique,
            so two functions in two different modules cannot be decorated with @step if they have the same name.
        """
        step_name = step_metadata.name

        if step_name in Pipeline.STEPS_REGISTRY:
            raise ValueError(
                f"More than one step is called '{step_name}'. The step names must be unique."
            )

        Pipeline.STEPS_REGISTRY[step_name] = step_metadata


@overload
def pipeline(_func: F) -> Pipeline:  # pragma: no cover
    ...


@overload
def pipeline(
    *,
    context: Optional[Any] = None,
    name: Optional[str] = None,
    log_folder_path: Optional[str] = None,
    remove_logs_on_completion: bool = True,
) -> Callable[[F], Pipeline]:  # pragma: no cover
    ...


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
