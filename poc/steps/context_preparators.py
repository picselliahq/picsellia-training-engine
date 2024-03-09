from picsellia import Client

from poc.step import step


@step
def context_preparator(
    api_token: str, host: str, organization_name: str, experiment_id: str
):
    client = Client(api_token=api_token, host=host, organization_name=organization_name)
    experiment = client.get_experiment_by_id(experiment_id)
    parameters = {
        "epochs": 1,
        "batch": 4,
        "imgsz": 224,
        "device": "mps",
        "cache": "ram",
    }
    context = {"client": client, "experiment": experiment, "parameters": parameters}
    return context
