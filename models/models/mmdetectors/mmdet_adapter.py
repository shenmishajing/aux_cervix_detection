import os
import shutil
from abc import ABC

import cv2
import mmcv
import numpy as np
import torch
from mmdet.core.visualization import imshow_gt_det_bboxes
from torch import nn
from torchmetrics.detection import MeanAveragePrecision

from models.models.base import LightningModule


class Timer:
    duration_ms = None

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.duration_ms = self.start.elapsed_time(self.end)

    @property
    def duration(self):
        return self.duration / 1000

    def iters_per_second(self, batch_size=1):
        return 1000 / self.duration_ms * batch_size


class MMDetModelAdapter(LightningModule, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(
        self,
        model: nn.Module,
        metrics: nn.ModuleList = None,
        metrics_log_info=None,
        imshow_kwargs=None,
        *args,
        **kwargs,
    ):
        """
        To show a metric in the progressbar a list of tupels can be provided for metrics_log_info, the first
        entry has to be the name of the metric to log and the second entry the display name in the progressbar. By default the
        mAP is logged to the progressbar.
        """
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.model = model
        if metrics is None:
            metrics, metrics_log_info = self.get_default_metrics()
        if len(metrics) > len(metrics_log_info):
            metrics_log_info += [
                {} for _ in range(len(metrics) - len(metrics_log_info))
            ]
        self.metrics = metrics
        self.metrics_log_info = metrics_log_info

        if imshow_kwargs is None:
            self.imshow_kwargs = {"score_thr": 0.5}
        else:
            self.imshow_kwargs = imshow_kwargs

    def get_default_metrics(self):
        metrics = nn.ModuleList([MeanAveragePrecision(class_metrics=True)])
        metrics_log_info = [{"prog_bar": [("map_50", "mAP")]}]
        return metrics, metrics_log_info

    def setup(self, stage=None):
        self.get_dataset()

    def get_dataset(self):
        self.dataset = self.trainer.datamodule.dataset

    def update_metrics(self, preds, target):
        for metric in self.metrics:
            metric.update(preds, target)

    def compute_metrics(self, prefix="val") -> None:
        for metric, metric_log_info in zip(self.metrics, self.metrics_log_info):
            metric_logs = metric.compute()

            if isinstance(metric, MeanAveragePrecision):
                metric_logs = dict(metric_logs)
                labels = [int(c) for c in metric._get_classes()]
                for k in [k for k in metric_logs if k.endswith("per_class")]:
                    res = metric_logs.pop(k)
                    if metric.class_metrics and res.ndim and len(labels) > 1:
                        for i in range(len(res)):
                            name = self.dataset.coco.loadCats(
                                self.dataset.cat_ids[labels[i]]
                            )[0]["name"]
                            metric_logs[k.replace("per_class", name)] = res[i]
            elif not isinstance(metric_logs, dict):
                if "log_name" in metric_log_info:
                    metric_logs = {metric_log_info["log_name"]: metric_logs}
                else:
                    metric_logs = {str(metric).removesuffix("()"): metric_logs}

            for k, v in metric_logs.items():
                self.log(f"{prefix}/{k}", v, sync_dist=True)

            if "prog_bar" in metric_log_info:
                if not isinstance(metric_log_info["prog_bar"], list):
                    metric_log_info["prog_bar"] = [metric_log_info["prog_bar"]]
                for item in metric_log_info["prog_bar"]:
                    key = value = None
                    if not isinstance(item, tuple) and len(metric_logs) == 1:
                        key = item
                        value = list(metric_logs.values())[0]
                    elif len(metric_logs) > 1:
                        if not isinstance(item, tuple):
                            item = (item, item)
                        if item[0] in metric_logs:
                            key = item[0]
                            value = metric_logs[key]
                    if key is not None:
                        self.log(
                            key, value, logger=False, prog_bar=True, sync_dist=True
                        )
            metric.reset()

    @staticmethod
    def unpack_preds(preds):
        stack_preds = torch.cat(preds)

        scores = stack_preds[:, -1]
        boxes = stack_preds[:, :-1]

        # each item in raw_pred is an array of predictions of it's `i` class
        labels = torch.cat(
            [stack_preds.new_full((p.shape[0],), i) for i, p in enumerate(preds)]
        )

        return {"boxes": boxes, "scores": scores, "labels": labels}

    def convert_raw_predictions(self, batch, preds):
        """Convert raw predictions from the model to library standard."""
        preds = [self.unpack_preds(p) for p in preds]
        target = [
            {"boxes": batch["gt_bboxes"][i], "labels": batch["gt_labels"][i]}
            for i in range(len(preds))
        ]
        return preds, target

    def forward(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)

    def forward_step(self, batch):
        self.batch_size = batch["img"].shape[0]

        outputs = self.model.train_step(data=batch, optimizer=None)
        with Timer() as timer:
            preds = self.model.simple_test(**batch)
        outputs["log_vars"]["iters_per_second"] = timer.iters_per_second(
            self.batch_size
        )

        preds, target = self.convert_raw_predictions(batch, preds)
        self.update_metrics(preds, target)
        return outputs

    def training_step(self, batch, batch_idx):
        self.batch_size = batch["img"].shape[0]
        outputs = self.model.train_step(data=batch, optimizer=None)
        self.log_dict(self.add_prefix(outputs["log_vars"]))
        return outputs["loss"]

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward_step(batch)
        self.log_dict(
            self.add_prefix(outputs["log_vars"], prefix="val/"), sync_dist=True
        )
        return outputs

    def validation_epoch_end(self, outs):
        self.compute_metrics()

    def test_step(self, batch, *args, **kwargs):
        outputs = self.forward_step(batch)
        self.log_dict(
            self.add_prefix(outputs["log_vars"], prefix="test/"), sync_dist=True
        )
        return outputs

    def test_epoch_end(self, outs):
        self.compute_metrics("test")

    @staticmethod
    def rm_and_create(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    @staticmethod
    def get_output_paths():
        return ["cam", "result"]

    def on_predict_start(self) -> None:
        log_dir = os.path.dirname(os.path.dirname(self.trainer.ckpt_path))
        self.output_path = os.path.join(log_dir, "visualization")

        output_paths = self.get_output_paths()
        for name in output_paths:
            path = os.path.join(self.output_path, name)
            self.rm_and_create(path)
            self.__setattr__(name + "_output_path", path)

    @staticmethod
    def denormalize_img(img, img_norm_cfg):
        return mmcv.rgb2bgr(img * img_norm_cfg["std"] + img_norm_cfg["mean"]).astype(
            np.uint8
        )

    def cam_visualization(self, img, img_metas=None, **kwargs):
        x = [
            xx.mean(dim=1, keepdim=True)
            for xx in self.model.extract_feat(img, stage="backbone")
        ]
        x = [
            nn.functional.interpolate(xx, size=img.shape[2:], mode="bilinear")
            for xx in x
        ]
        x = [
            ((xx - xx.min()) / (xx.max() - xx.min(xx)) * 255 + 0.5).clamp_(0, 255)
            for xx in x
        ]
        x = [xx.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy() for xx in x]

        imgs = img.permute(0, 2, 3, 1).cpu().numpy()
        for i in range(len(img_metas)):
            for layer_num, xx in enumerate(x):
                mmcv.imwrite(
                    cv2.applyColorMap(xx[i], cv2.COLORMAP_JET),
                    os.path.join(
                        self.cam_output_path,
                        os.path.splitext(img_metas[i]["ori_filename"])[0]
                        + f"_{layer_num}.png",
                    ),
                )
            mmcv.imwrite(
                self.denormalize_img(imgs[i], img_metas[i]["img_norm_cfg"]),
                os.path.join(
                    self.cam_output_path,
                    os.path.splitext(img_metas[i]["ori_filename"])[0] + ".png",
                ),
            )

    def result_visualization(self, batch, *args, **kwargs):
        preds = self.model.simple_test(**batch)
        imgs = batch["img"].permute(0, 2, 3, 1).cpu().numpy()
        for i in range(len(preds)):
            img = self.denormalize_img(imgs[i], batch["img_metas"][i]["img_norm_cfg"])
            ann = {
                "gt_bboxes": batch["gt_bboxes"][i].cpu().numpy(),
                "gt_labels": batch["gt_labels"][i].cpu().numpy(),
            }
            pred = [bbox.cpu().numpy() for bbox in preds[i]]
            imshow_gt_det_bboxes(
                img,
                ann,
                pred,
                class_names=self.dataset.CLASSES,
                show=False,
                **self.imshow_kwargs,
                out_file=os.path.join(
                    self.result_output_path, batch["img_metas"][i]["ori_filename"]
                ),
            )

    def predict_step(self, batch, *args, **kwargs):
        self.cam_visualization(**batch)
        self.result_visualization(batch, *args, **kwargs)
