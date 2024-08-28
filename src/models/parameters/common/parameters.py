import logging
from abc import ABC
from typing import Any, Dict, Optional, Set, Tuple, TypeVar, get_origin, get_args, Union

from picsellia.types.schemas import LogDataType  # type: ignore

from src import Colors

logger = logging.getLogger("picsellia-engine")


class Parameters(ABC):
    def __init__(self, log_data: LogDataType):
        self.parameters_data = self.validate_log_data(log_data)
        self.defaulted_keys: Set[str] = set()

    def extract_parameter(
        self,
        keys: list,
        expected_type: type,
        default: Any = ...,
        range_value: Optional[Tuple[Any, Any]] = None,
    ) -> Any:
        """Extract a parameter from the log data.

        This function tries to extract a parameter from the log data using a list of possible keys.
        Additional constraints can be provided, such as the expected type, a default value, and a value range.

        Examples:
            Extract a required string parameter that cannot be None:
            ```
            parameter = self.extract_parameter(keys=["key1", "key2"], expected_type=str)
            ```

            Extract a required integer parameter that can be None:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=int | None)
            ```

            Extract an optional float parameter within a specific range:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=float, default=0.5, range_value=(0.0, 1.0))
            ```

            Extract an optional string parameter with a default value:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=str, default="default_value")
            ```

            Extract an optional string parameter that can be None:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=Union[str, None], default=None)
            ```

        Args:
            keys: A list of possible keys to extract the parameter.
            expected_type: The expected type of the parameter, can use Union for optional types.
            default: The default value if the parameter is not found. Use ... for required parameters.
            range_value: A tuple of two numbers representing the allowed range of the parameter.

        Returns:
            The extracted parameter.

        Raises:
            ValueError: If no keys are provided or if the value is out of the allowed range.
            TypeError: If the parameter is not of the expected type.
            KeyError: If no parameter is found and no default value is provided.
        """

        if len(keys) == 0:
            raise ValueError(
                "Cannot extract a parameter without any keys. One or more keys must be provided."
            )

        # Determine if the type is optional
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        is_optional = origin is Union and any(isinstance(None, arg) for arg in args)
        base_type = (
            next((arg for arg in args if not isinstance(None, arg)), expected_type)
            if is_optional
            else expected_type
        )

        # Check if the default value matches the expected type
        if default is not ... and not isinstance(default, (base_type, type(None))):
            raise TypeError(
                f"The provided default value {default} does not match the expected type {expected_type}."
            )

        if default is None and not is_optional:
            raise ValueError(
                f"The default value cannot be None as the expected type {expected_type} is not optional."
            )

        for key in keys:
            if key in self.parameters_data:
                value = self.parameters_data[key]
                parsed_value = self._flexible_type_check(
                    value, base_type, is_optional=is_optional
                )

                if parsed_value is None and not is_optional:
                    raise TypeError(
                        f"The value for key '{key}' cannot be None as it's not an optional type."
                    )

                if parsed_value is not None:
                    if range_value and base_type in [int, float]:
                        checked_value_range = self._validate_range(range_value)
                        if not (
                            checked_value_range[0]
                            <= parsed_value
                            <= checked_value_range[1]
                        ):
                            raise ValueError(
                                f"Value for key '{key}' is out of the allowed range {range_value}."
                            )
                    return parsed_value

                elif is_optional:
                    return parsed_value

                else:
                    raise RuntimeError(
                        f"The value {value} for key {key} has been parsed to None and therefore cannot be used. "
                        f"The key {key} expects a value of type {expected_type}."
                    )

        if default is not ...:
            logger.warning(
                f"None of the keys {keys} were found in the provided data. "
                f"Using default value {Colors.YELLOW}{default}{Colors.ENDC}."
            )
            self.defaulted_keys.update(keys)
            return default

        else:
            error_string = f"Required parameter with key(s) {keys} of type {expected_type} not found."

            if range_value is not None:
                error_string += f" Expected value within the range {range_value}."

            raise KeyError(error_string)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameters to a dictionary.

        This function gathers all of its parameters and returns them as a dictionary.
        Some parameters are excluded from the dictionary, such as `parameters_data` and `defaulted_keys`.

        Returns:
            The parameters as a dictionary.
        """
        filtered_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["parameters_data", "defaulted_keys"]
        }
        return dict(sorted(filtered_dict.items()))

    def validate_log_data(self, log_data: LogDataType) -> Dict[str, Any]:
        """Validate the log data.

        Args:
            log_data: The log data to validate.

        Returns:
            The validated log data.

        Raises:
            ValueError: If the log data is not a dictionary.
        """
        if isinstance(log_data, dict):
            return log_data

        raise ValueError("The provided parameters must be a dictionary.")

    def _flexible_type_check(
        self, value: Any, expected_type: type, is_optional: bool
    ) -> Any:
        """Check if a value can be converted to a given type.

        Args:
            value: The value to check.
            expected_type: The type to check against.
            is_optional: Whether the type is optional.

        Returns:
            The value converted to the expected type if possible, otherwise None.

        Raises:
            ValueError: If the value cannot be converted to the expected type.
        """
        if expected_type is bool:
            if isinstance(value, bool):
                return value
            if str(value).lower() in ["1", "true", "yes"]:
                return True
            if str(value).lower() in ["0", "false", "no"]:
                return False

            raise ValueError(
                f"Value {value} cannot be converted to a boolean."
                f"Accepted values are '1', 'true', 'yes', '0', 'false', 'no'."
            )

        elif expected_type is float:
            if isinstance(value, (int, float)):
                return float(value)  # Directly converts int to float or maintains float
            try:
                return float(str(value))  # Attempt to convert string to float
            except ValueError as e:
                raise ValueError(f"Value {value} cannot be converted to float.") from e

        elif expected_type is int:
            if isinstance(value, int):
                return value  # No conversion needed
            elif isinstance(value, float):
                if value.is_integer():
                    return int(value)  # Convert to int if it's a whole number
                else:
                    raise ValueError(
                        f"Value {value} cannot be converted to int without losing precision."
                    )
            else:
                try:
                    # Attempt to convert string to float first to handle cases like "100.0"
                    float_value = float(str(value))
                    if float_value.is_integer():
                        return int(float_value)  # Convert to int if it's a whole number
                    else:
                        raise ValueError
                except ValueError as e:
                    raise ValueError(
                        f"Value {value} cannot be converted to int without losing precision."
                    ) from e

        elif value is None and not is_optional:
            raise TypeError(
                f"Value {value} cannot be None as it's not an optional type."
            )

        elif is_optional:
            if value is None:
                return value
            elif str(value).lower() in ["none", "null"]:
                return None

        return value

    def _validate_range(self, value_range: tuple) -> tuple:
        """Validate the range of a value.

        Args:
            value_range: A list of two numbers representing the range.

        Returns:
            The validated range.

        Raises:
            ValueError: If the range is invalid.
        """
        if value_range is not None:
            if (
                len(value_range) == 2
                and isinstance(value_range[0], (int, float))
                and isinstance(value_range[1], (int, float))
                and value_range[0] < value_range[1]
            ):
                return value_range

        raise ValueError(
            f"The provided range value {value_range} is invalid. "
            "It must be a tuple of two numbers (int or float) "
            "where the first is strictly less than the second."
        )


TParameters = TypeVar("TParameters", bound=Parameters)
