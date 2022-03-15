import csv
import os
from abc import ABC

import mmcv
import torch
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

    def on_predict_start(self) -> None:
        super().on_predict_start()
        self.correct_output_path = os.path.join(self.output_path, 'correct')
        self.wrong_output_path = os.path.join(self.output_path, 'wrong')
        os.makedirs(self.correct_output_path)
        os.makedirs(self.wrong_output_path)
        self.correct_preds = []
        self.wrong_preds = []

    def predict_step(self, batch, *args, **kwargs):
        preds = self.model.simple_test(**batch)
        preds, target = self.convert_raw_predictions(batch, preds)
        pred_inds = torch.argmax(preds, dim = 1)
        result = pred_inds == target
        imgs = batch['img'].permute(0, 2, 3, 1).cpu().numpy()
        for i, res in enumerate(result):
            img = self.denormalize_img(imgs[i], batch['img_metas'][i]['img_norm_cfg'])
            output_path = self.correct_output_path if res else self.wrong_output_path
            output_path = os.path.join(output_path, f'gt={target[i]}_pred={pred_inds[i]}_' + batch['img_metas'][i]['ori_filename'])
            if not os.path.exists(os.path.basename(output_path)):
                os.makedirs(os.path.dirname(output_path))
            mmcv.imwrite(img, output_path)
            pred_list = self.correct_preds if res else self.wrong_preds
            pred_list.append([res.item(), target[i].item(), pred_inds[i].item(), *preds[i].cpu().numpy().tolist(),
                              batch['img_metas'][i]['ori_filename']])

    def on_predict_end(self) -> None:
        preds = self.wrong_preds + self.correct_preds
        if len(preds):
            file_header = ['correct', 'gt', 'pred'] + [f'score_{i}' for i in range(len(preds[0]) - 4)] + ['filename']
            with open(os.path.join(self.output_path, 'preds.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(file_header)
                writer.writerows(preds)
