import logging
import os

from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.yolo.v8.segment.train import SegmentationTrainer
from utils import send_run_to_picsellia


class PicselliaSegmentationTrainer(SegmentationTrainer):
    def __init__(self, experiment: Experiment, cfg=None, parameters: dict = None):
        args = dict(
            task=cfg.task,
            model=cfg.model or "yolov8n-seg.pt",
            data=cfg.data or "coco128-seg.yaml",
            device=cfg.device if cfg.device is not None else "",
            epochs=cfg.epochs,
            batch=cfg.batch,
            imgsz=cfg.imgsz,
            save_period=cfg.save_period,
            name=cfg.name,
            project=cfg.project,
            patience=cfg.patience,
        )
        cfg_dict = vars(cfg)
        for parameter_key in parameters.keys():
            if parameter_key in cfg_dict:
                parameter_type = type(cfg_dict[parameter_key])
                args[parameter_key] = parameter_type(parameters[parameter_key])

        print("here are args ", args)
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
                model.export(format="onnx", imgsz=self.args.imgsz, task="segment")
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
