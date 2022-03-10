from abc import ABC

from torch import nn
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, Specificity

from ..mmdetectors import MMDetModelAdapter


class MMClsModelAdapter(MMDetModelAdapter, ABC):
    def get_default_metrics(self):
        metrics = nn.ModuleList([
            Accuracy(),
            Precision(),
            Recall(),
            F1Score(),
            Specificity()
        ])
        metrics_log_info = [
            {'prog_bar': ['acc']},
        ]
        return metrics, metrics_log_info

    def convert_raw_predictions(self, batch, preds):
        """Convert raw predictions from the model to library standard."""
        return preds, batch['gt_label']

    def predict_step(self, batch, *args, **kwargs):
        pass
