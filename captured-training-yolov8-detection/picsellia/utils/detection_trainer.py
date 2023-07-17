import logging
import os

from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from utils.processing import send_run_to_picsellia


class PicselliaDetectionTrainer(DetectionTrainer):
    def __init__(self, experiment: Experiment, cfg=None):
        task = cfg.task
        model = cfg.model or "yolov8n.pt"
        data = cfg.data or "coco128.yaml"  # or yolo.ClassificationDataset("mnist")
        device = cfg.device if cfg.device is not None else ""
        epochs = cfg.epochs
        batch = cfg.batch
        imgsz = cfg.imgsz
        save_period = cfg.save_period
        save_dir = cfg.cwd
        project = cfg.project
        patience = cfg.patience

        args = dict(
            task=task,
            model=model,
            data=data,
            device=device,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            save_period=save_period,
            project=project,
            patience=patience,
        )
        super().__init__(overrides=args)
        self.experiment = experiment
        self.cwd = cfg.cwd

    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        for name, value in metrics.items():
            log_name = str(name).replace("/", "_")
            self._log_metric(log_name, float(value), retry=1)

        if (
            (self.epoch > 0)
            and (self.save_period > 0)
            and ((self.epoch - 1) % self.save_period == 0)
        ):
            model = YOLO(os.path.join(self.save_dir, "weights", "best.pt"))
            model.export(format="onnx", imgsz=self.args.imgsz, task="detect")
            send_run_to_picsellia(
                experiment=self.experiment, cwd=self.cwd, save_dir=self.save_dir
            )

    def _log_metric(self, name: str, value: float, retry: int):
        try:
            self.experiment.log(name=name, type=LogType.LINE, data=value)
        except Exception:
            logging.exception(f"couldn't log {name}")
            if retry > 0:
                logging.info(f"retrying log {name}")
                self._log_metric(name, value, retry - 1)
