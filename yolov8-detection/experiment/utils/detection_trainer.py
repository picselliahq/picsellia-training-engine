import logging
import os

from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from utils.processing import send_run_to_picsellia


class PicselliaDetectionTrainer(DetectionTrainer):
    def __init__(self, experiment: Experiment, cfg=None):
        excluded_keys = {"cwd", "mode", "task"}
        args = {
            key: value for key, value in vars(cfg).items() if key not in excluded_keys
        }
        args.setdefault("model", "yolov8n.pt")
        args.setdefault("data", "coco128.yaml")
        args.setdefault("device", "")
        super().__init__(overrides=args)
        self.experiment = experiment
        self.cwd = cfg.cwd

    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        model = None
        for name, value in metrics.items():
            log_name = str(name).replace("/", "_")
            self._log_metric(log_name, float(value), retry=1)
        if (
            (self.epoch > 1)
            and (self.save_period > 0)
            and ((self.epoch - 1) % self.save_period == 0)
        ):
            try:
                model = YOLO(os.path.join(self.save_dir, "weights", "best.pt"))
            except FileNotFoundError:
                try:
                    model = YOLO(os.path.join(self.save_dir, "weights", "last.pt"))
                except FileNotFoundError as e:
                    logging.warning(
                        "Can't find intermediary checkpoint at save period, they will be uploaded at the end"
                    )
                    logging.warning(e)
            if model:
                model.export(format="onnx", imgsz=self.args.imgsz, task="detect")
                send_run_to_picsellia(
                    experiment=self.experiment, cwd=self.cwd, save_dir=self.save_dir
                )

    def _log_metric(self, name: str, value: float, retry: int):
        try:
            self.experiment.log(name=name, type=LogType.LINE, data=value)
        except Exception as e:
            logging.exception(f"couldn't log {name}")
            logging.warning(e)
            if retry > 0:
                logging.info(f"retrying log {name}")
                self._log_metric(name, value, retry - 1)
