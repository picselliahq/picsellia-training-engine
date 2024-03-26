import logging
from abc import ABC
from typing import Any, Set, Optional, Tuple

from picsellia.types.schemas import LogDataType  # type: ignore


logger = logging.getLogger("picsellia")


class Parameters(ABC):
    def __init__(self, log_data: LogDataType):
        self.parameters_data = self.validate_log_data(log_data)
        self.defaulted_keys: Set[str] = set()

    def extract_parameter(
        self,
        keys: list,
        expected_type: type,
        default: Any = None,
        value_range: Optional[Tuple[Any, Any]] = None,
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
            parameter = self.extract_parameter(keys=["key1"], expected_type=float, value_range=[0.0, 1.0])
            ```

        Args:
            keys: A list of possible keys to extract the parameter.
            expected_type: The expected type of the parameter.
            default: The default value if the parameter is not found.
            value_range: A tuple of two numbers representing the allowed range of the parameter.

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

        for key in keys:
            if key in self.parameters_data:
                value = self.parameters_data[key]
                checked_value = self._flexible_type_check(value, expected_type)

                if checked_value is not None:
                    if value_range and expected_type in [int, float]:
                        checked_value_range = self._validate_range(value_range)
                        if not (
                            checked_value_range[0]
                            < checked_value
                            < checked_value_range[1]
                        ):
                            raise ValueError(
                                f"Value for key '{key}' is out of the allowed range {value_range}."
                            )
                    return checked_value
                else:
                    raise TypeError(
                        f"Value for key '{key}' is not of expected type `{expected_type.__name__}`."
                    )

        if default is not None:
            logger.warning(
                f"None of the keys {keys} were found in the provided data. "
                f"Using default value \033[33m{default}\033[0m."
            )
            self.defaulted_keys.update(keys)
            return self._flexible_type_check(default, expected_type)

        else:
            error_string = (
                f"Some parameters are missing. "
                f"At least one parameter with a key from {keys}"
            )

            if value_range is not None:
                error_string += f", of type `{expected_type.__name__}` and within the range {value_range}"
            else:
                error_string += f" and of type `{expected_type.__name__}`"

            error_string += " must be provided."

            raise KeyError(error_string)

    def to_dict(self) -> dict:
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

    def validate_log_data(self, log_data: LogDataType) -> dict:
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
            return None  # If the value doesn't match any known boolean representations

        elif expected_type is float:
            if isinstance(value, (int, float)):
                return float(value)  # Directly converts int to float or maintains float
            else:
                return None  # Return None if conversion isn't straightforward

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
            if not (
                len(value_range) == 2
                and isinstance(value_range[0], (int, float))
                and isinstance(value_range[1], (int, float))
                and value_range[0] < value_range[1],
            ):
                raise ValueError(
                    f"The provided range value {value_range} is invalid. "
                    "It must be a tuple of two numbers (int or float) "
                    "where the first is strictly less than the second."
                )
        return value_range
