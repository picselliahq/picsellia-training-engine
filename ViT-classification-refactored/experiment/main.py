import logging
import os
from transformers import TrainerCallback
from trainer import VitClassificationTrainer

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger("picsellia").setLevel(logging.INFO)


class LogMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            for metric_name, value in logs.items():
                if metric_name in [
                    "train_loss",
                    "total_flos",
                    "train_steps_per_second",
                    "train_samples_per_second",
                    "train_runtime",
                ]:
                    experiment.log(str(metric_name), float(value), "value")
                else:
                    experiment.log(str(metric_name), float(value), "line")


training_pipeline = VitClassificationTrainer()
training_pipeline.prepare_data_for_training()
experiment = training_pipeline.experiment
training_pipeline.init_train()
training_pipeline.trainer.add_callback(LogMetricsCallback)
training_pipeline.train()
training_pipeline.eval()
