import logging
from abc import ABC
from typing import Any, Dict, Optional, Set, Tuple

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
        default: Any = None,
        range_value: Optional[Tuple[Any, Any]] = None,
    ) -> Any:
        """Extract a parameter from the log data.

        This function tries to extract a parameter from the log data using a list of possible keys.
        Additional constraints can be provided, such as the expected type, a default value, and a value range.

        Examples:
            Extract a string parameter that can either have the key "key1" or "key2":
            ```
            parameter = self.extract_parameter(keys=["key1", "key2"], expected_type=str)
            ```

            Extract an integer parameter with a default value:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=int, default=42)
            ```

            Extract a float parameter within a specific range:
            ```
            parameter = self.extract_parameter(keys=["key1"], expected_type=float, range_value=[0.0, 1.0])
            ```

        Args:
            keys: A list of possible keys to extract the parameter.
            expected_type: The expected type of the parameter.
            default: The default value if the parameter is not found.
            range_value: A tuple of two numbers representing the allowed range of the parameter.

        Returns:
            The extracted parameter.

        Raises:
            ValueError: If no keys are provided.
            ValueError: If the value is out of the allowed range.
            TypeError: If the parameter is not of the expected type.
            KeyError: If no parameter is found and no default value is provided.
        """
        if len(keys) == 0:
            raise ValueError(
                "Cannot extract a parameter without any keys. One or more keys must be provided."
            )

        if default is not None and not isinstance(default, expected_type):
            raise TypeError(
                f"The provided default value {default} does not match the expected type {expected_type}."
            )

        for key in keys:
            if key in self.parameters_data:
                value = self.parameters_data[key]
                parsed_value = self._flexible_type_check(value, expected_type)

                if parsed_value is not None:
                    if range_value and expected_type in [int, float]:
                        checked_value_range = self._validate_range(range_value)
                        if not (
                            checked_value_range[0]
                            < parsed_value
                            < checked_value_range[1]
                        ):
                            raise ValueError(
                                f"Value for key '{key}' is out of the allowed range {range_value}."
                            )
                    return parsed_value
                else:
                    raise RuntimeError(
                        f"The value {value} for key {key} has been parsed to None and therefore cannot be used. "
                        f"The key {key} except as value of type {expected_type}."
                    )

        if default is not None:
            logger.warning(
                f"None of the keys {keys} were found in the provided data. "
                f"Using default value {Colors.YELLOW}{default}{Colors.ENDC}."
            )
            self.defaulted_keys.update(keys)
            return self._flexible_type_check(default, expected_type)

        else:
            error_string = (
                f"Some parameters are missing. "
                f"At least one parameter with a key from {keys}"
            )

            if range_value is not None:
                error_string += f", of type `{expected_type.__name__}` and within the range {range_value}"
            else:
                error_string += f" and of type `{expected_type.__name__}`"

            error_string += " must be provided."

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

    def _flexible_type_check(self, value: Any, expected_type: type) -> Any:
        """Check if a value can be converted to a given type.

        Args:
            value: The value to check.
            expected_type: The type to check against.

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
