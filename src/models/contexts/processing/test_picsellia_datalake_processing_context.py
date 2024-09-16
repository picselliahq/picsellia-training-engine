from typing import Any, Dict, Optional, Union, List
from uuid import UUID

import orjson

from src.models.contexts.common.picsellia_context import PicselliaContext
from picsellia.types.enums import ProcessingType
from picsellia import ModelVersion, Datalake, Client, Job


def create_processing(
    client: Client,
    name: str,
    type: Union[str, ProcessingType],
    default_cpu: int,
    default_gpu: int,
    default_parameters: dict,
    docker_image: str,
    docker_tag: str,
    docker_flags: Optional[List[str]] = None,
) -> str:
    payload = {
        "name": name,
        "type": type,
        "default_cpu": default_cpu,
        "default_gpu": default_gpu,
        "default_parameters": default_parameters,
        "docker_image": docker_image,
        "docker_tag": docker_tag,
        "docker_flags": docker_flags,
    }
    r = client.connexion.post(
        f"/sdk/organization/{client.id}/processings", data=orjson.dumps(payload)
    ).json()
    return r["id"]


def get_processing(client: Client, name: str) -> str:
    r = client.connexion.get(
        f"/sdk/organization/{client.id}/processings", params={"name": name}
    ).json()
    return r["items"][0]["id"]


def launch_processing(
    client: Client,
    datalake: Datalake,
    data_ids: list[UUID],
    model_version_id: str,
    processing_id: str,
    parameters: dict,
    cpu: int,
    gpu: int,
    target_datalake_name: Optional[str] = None,
):
    payload = {
        "processing_id": processing_id,
        "parameters": parameters,
        "cpu": cpu,
        "gpu": gpu,
        "model_version_id": model_version_id,
        "data_ids": data_ids,
    }

    if target_datalake_name:
        payload["target_datalake_name"] = target_datalake_name

    r = client.connexion.post(
        f"/api/datalake/{datalake.id}/processing/launch",
        data=orjson.dumps(payload),
    ).json()
    return Job(client.connexion, r, version=2)


class TestPicselliaDatalakeProcessingContext(PicselliaContext):
    """
    This class is used to test a processing pipeline without a real job execution on Picsellia (without giving a real job ID).
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        organization_id: Optional[str] = None,
        job_id: Optional[str] = None,
        job_type: Optional[str] = None,
        input_datalake_id: Optional[str] = None,
        output_datalake_id: Optional[str] = None,
        model_version_id: Optional[str] = None,
        offset: Optional[int] = 0,
        limit: Optional[int] = 100,
        use_id: Optional[bool] = True,
        processing_parameters=None,
    ):
        # Initialize the Picsellia client from the base class
        super().__init__(api_token, host, organization_id)

        self.job_id = job_id
        self.job_type = job_type

        self.input_datalake_id = input_datalake_id
        self.output_datalake_id = output_datalake_id
        self.model_version_id = model_version_id

        if not input_datalake_id:
            raise ValueError("Input datalake ID must be provided")
        self.input_datalake = self.get_datalake(input_datalake_id)
        if not self.input_datalake:
            raise ValueError(f"Datalake with ID {input_datalake_id} not found")
        self.output_datalake = self.get_datalake(output_datalake_id)
        self.model_version = self.get_model_version(model_version_id=model_version_id)

        self.offset = offset
        self.limit = limit
        self.data_ids = self.get_data_ids(
            datalake=self.input_datalake, offset=self.offset, limit=limit
        )

        self.use_id = use_id
        self.processing_parameters = processing_parameters

    def get_datalake(self, datalake_id: str) -> Optional[Datalake]:
        if not datalake_id:
            return None
        return self.client.get_datalake(id=datalake_id)

    def get_model_version(self, model_version_id: str) -> Optional[ModelVersion]:
        if not model_version_id:
            return None
        return self.client.get_model_version_by_id(model_version_id)

    def get_data_ids(self, datalake: Datalake, offset: int, limit: int) -> list[UUID]:
        if not datalake or offset is None or limit is None:
            raise (ValueError("Datalake, offset and limit must be provided"))
        else:
            return datalake.list_data(offset=offset, limit=limit).ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "job_type": self.job_type,
                "input_datalake_id": self.input_datalake_id,
                "output_datalake_id": self.output_datalake_id,
                "model_version_id": self.model_version_id,
                "offset": self.offset,
                "limit": self.limit,
                "use_id": self.use_id,
            },
            "processing_parameters": self.processing_parameters,
        }
