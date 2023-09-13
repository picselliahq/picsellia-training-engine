from abc import ABC, abstractmethod
from picsellia import Experiment, Client
import os


class AbstractTrainer(ABC):
    def __init__(self):
        self.experiment = self.get_experiment()
        self.dataset_list = self.experiment.list_attached_dataset_versions()
        self.parameters = self.experiment.get_log("parameters").data
        self.labelmap = {}

    @staticmethod
    def get_experiment() -> Experiment:
        if "api_token" not in os.environ:
            raise Exception("You must set an api_token to run this image")
        api_token = os.environ["api_token"]

        if "host" not in os.environ:
            host = "https://app.picsellia.com"
        else:
            host = os.environ["host"]

        if "organization_name" not in os.environ:
            organization_name = None
        else:
            organization_name = os.environ["organization_name"]

        client = Client(
            api_token=api_token, host=host, organization_name=organization_name
        )
        if "experiment_id" in os.environ:
            experiment_id = os.environ["experiment_id"]

            experiment = client.get_experiment_by_id(experiment_id)
        else:
            raise Exception("You must set the experiment_id")
        return experiment

    def prepare_data_for_training(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass
