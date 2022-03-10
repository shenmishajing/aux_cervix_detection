import os
import shutil
from abc import ABC
from typing import List, Union

import mmcv
import numpy as np
import torch
from mmdet.core.visualization import imshow_gt_det_bboxes
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.metric import Metric

from models.models.base import LightningModule


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
            metrics: List[Metric] = None,
            metrics_keys_to_log_to_prog_bar: List[Union[str, tuple]] = None,
            imshow_kwargs = None,
            *args, **kwargs
    ):
        """
        To show a metric in the progressbar a list of tupels can be provided for metrics_keys_to_log_to_prog_bar, the first
        entry has to be the name of the metric to log and the second entry the display name in the progressbar. By default the
        mAP is logged to the progressbar.
        """
        super().__init__(*args, **kwargs)
        self.dataset = None
        self.model = model
        self.metrics = metrics or [MeanAveragePrecision(class_metrics = True)]
        self.metrics_keys_to_log_to_prog_bar = metrics_keys_to_log_to_prog_bar or [('map_50', 'mAP')]
        for i in range(len(self.metrics_keys_to_log_to_prog_bar)):
            if isinstance(self.metrics_keys_to_log_to_prog_bar[i], str):
                self.metrics_keys_to_log_to_prog_bar[i] = (self.metrics_keys_to_log_to_prog_bar[i], self.metrics_keys_to_log_to_prog_bar[i])

        if imshow_kwargs is None:
            self.imshow_kwargs = {'score_thr': 0.5}
        else:
            self.imshow_kwargs = imshow_kwargs

    def setup(self, stage = None):
        self.get_dataset()

    def get_dataset(self):
        self.dataset = self.trainer.datamodule.dataset

    def update_metrics(self, preds, target):
        for metric in self.metrics:
            metric.update(preds, target)

    def compute_metrics(self, prefix = 'val') -> None:
        for metric in self.metrics:
            metric_logs = metric.compute()
            if isinstance(metric, MeanAveragePrecision):
                metric_logs = dict(metric_logs)
                labels = [int(c) for c in metric._get_classes()]
                for k in [k for k in metric_logs if k.endswith('per_class')]:
                    res = metric_logs.pop(k)
                    if metric.class_metrics and res.ndim and len(labels) > 1:
                        for i in range(len(res)):
                            name = self.dataset.coco.loadCats(self.dataset.cat_ids[labels[i]])[0]['name']
                            metric_logs[k.replace('per_class', name)] = res[i]
            elif not isinstance(metric_logs, dict):
                metric_logs = {str(metric).removesuffix('()'): metric_logs}
            for k, v in metric_logs.items():
                for entry in self.metrics_keys_to_log_to_prog_bar:
                    if entry[0] == k:
                        self.log(entry[1], v, logger = False, prog_bar = True)
                    self.log(f'{prefix}/{k}', v)
            metric.reset()

    @staticmethod
    def unpack_preds(preds):
        stack_preds = torch.cat(preds)

        scores = stack_preds[:, -1]
        boxes = stack_preds[:, :-1]

        # each item in raw_pred is an array of predictions of it's `i` class
        labels = torch.cat([stack_preds.new_full((p.shape[0],), i) for i, p in enumerate(preds)])

        return {'boxes': boxes, 'scores': scores, 'labels': labels}

    @staticmethod
    def convert_raw_predictions(batch, preds):
        """Convert raw predictions from the model to library standard."""
        preds = [MMDetModelAdapter.unpack_preds(p) for p in preds]
        target = [{'boxes': batch['gt_bboxes'][i], 'labels': batch['gt_labels'][i]} for i in range(len(preds))]
        return preds, target

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward_step(self, batch):
        self.batch_size = batch['img'].shape[0]
        with torch.no_grad():
            outputs = self.model.train_step(data = batch, optimizer = None)
            preds = self.model.simple_test(**batch)

        preds, target = self.convert_raw_predictions(batch, preds)
        self.update_metrics(preds, target)
        return outputs

    def training_step(self, batch, batch_idx):
        self.batch_size = batch['img'].shape[0]
        outputs = self.model.train_step(data = batch, optimizer = None)
        self.log_dict(self.add_prefix(outputs['log_vars']))
        return outputs['loss']

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.forward_step(batch)
        self.log_dict(self.add_prefix(outputs['log_vars'], prefix = 'val/'))
        return outputs

    def validation_epoch_end(self, outs):
        self.compute_metrics()

    def test_step(self, batch, *args, **kwargs):
        outputs = self.forward_step(batch)
        self.log_dict(self.add_prefix(outputs['log_vars'], prefix = 'test/'))
        return outputs

    def test_epoch_end(self, outs):
        self.compute_metrics('test')

    def on_predict_start(self) -> None:
        log_dir = os.path.dirname(os.path.dirname(self.trainer.predicted_ckpt_path))
        self.output_path = os.path.join(log_dir, 'visualization')
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

    @staticmethod
    def denormalize_img(img, img_norm_cfg):
        return mmcv.rgb2bgr(img * img_norm_cfg['std'] + img_norm_cfg['mean']).astype(np.uint8)

    def predict_step(self, batch, *args, **kwargs):
        preds = self.model.simple_test(**batch)
        imgs = batch['img'].permute(0, 2, 3, 1).cpu().numpy()
        for i in range(len(preds)):
            img = self.denormalize_img(imgs[i], batch['img_metas'][i]['img_norm_cfg'])
            ann = {'gt_bboxes': batch['gt_bboxes'][i].cpu().numpy(), 'gt_labels': batch['gt_labels'][i].cpu().numpy()}
            pred = [bbox.cpu().numpy() for bbox in preds[i]]
            imshow_gt_det_bboxes(img, ann, pred, class_names = self.dataset.CLASSES, show = False, **self.imshow_kwargs,
                                 out_file = os.path.join(self.output_path, batch['img_metas'][i]['ori_filename']))
