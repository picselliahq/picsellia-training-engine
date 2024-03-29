import logging
from abc import ABC

from picsellia.types.schemas import LogDataType


logger = logging.getLogger("picsellia")


class Parameters(ABC):
    def __init__(self, log_data: LogDataType):
        self.parameters_data = self.validate_log_data(log_data)
        self.defaulted_keys = set()

    def extract_parameter(
        self, keys: list, expected_type: type, default=None, value_range=None
    ):
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
        filtered_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["parameters_data", "defaulted_keys"]
        }
        return dict(sorted(filtered_dict.items()))

    def validate_log_data(self, log_data: LogDataType) -> dict:
        if isinstance(log_data, dict):
            return log_data

        raise ValueError("The provided parameters must be a dictionary.")

    def _flexible_type_check(self, value, expected_type):
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

    def _validate_range(self, value_range: list):
        if value_range is not None:
            if not (
                isinstance(value_range[0], (int, float))
                and isinstance(value_range[1], (int, float))
                and value_range[0] < value_range[1]
            ):
                raise ValueError(
                    f"The provided range value {value_range} is invalid. "
                    "It must be a tuple of two numbers (int or float) "
                    "where the first is strictly less than the second."
                )
        return value_range
