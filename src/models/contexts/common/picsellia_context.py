import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

import picsellia  # type: ignore


class PicselliaContext(ABC):
    def __init__(
        self,
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        organization_id: Optional[str] = None,
        organization_name: Optional[str] = None,
    ):
        self.api_token = api_token or os.getenv("api_token")

        if not self.api_token:
            raise ValueError(
                "API token not provided. Please provide it as an argument or set the 'api_token' environment variable."
            )

        self.host = host or os.getenv("host", "https://app.picsellia.com")
        self.organization_id = organization_id or os.getenv("organization_id")
        self.organization_name = organization_name or os.getenv("organization_name")

        self.client = self._initialize_client()

    def _initialize_client(self) -> picsellia.Client:
        return picsellia.Client(
            api_token=self.api_token,
            host=self.host,
            organization_id=self.organization_id,
            organization_name=self.organization_name,
        )

    def _format_parameter_with_color_and_suffix(
        self, value: Any, key: str, defaulted_keys: set
    ) -> str:
        """
        Formats a given value with ANSI color codes and a suffix indicating whether it is a default value.

        Args:
            value: The value to be formatted. This could be of any type but is converted to a string.
            key: The key associated with the value. Used to determine if the value is defaulted.
            defaulted_keys: A set of keys that are considered to have default values.

        Returns:
            The value formatted as a string with appropriate color coding and suffix.
        """
        suffix = " (default)" if key in defaulted_keys else ""
        color_code = "\033[33m" if suffix else "\033[34m"
        return f"{color_code}{value}\033[0m{suffix}"

    def _process_parameters(self, parameters_dict: Dict, defaulted_keys: set) -> Dict:
        """
        Processes parameters by applying color coding and suffixes to their values based on whether they are default.

        Args:
            parameters_dict: The dictionary of parameters to process.
            defaulted_keys: A set of parameter keys that are considered to have default values.

        Returns:
            A dictionary of processed parameters with color coding and suffixes applied.
        """
        processed_params = {}
        for key, value in parameters_dict.items():
            processed_params[key] = self._format_parameter_with_color_and_suffix(
                value, key, defaulted_keys
            )
        return processed_params

    @abstractmethod
    def to_dict(self):
        pass
