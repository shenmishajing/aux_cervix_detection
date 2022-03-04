import os
from abc import ABC

import mmcv
import numpy as np
from mmdet.core.visualization import imshow_gt_det_bboxes

from .mmdet_adapter import MMDetModelAdapter


class MMDetMultiModalsModelAdapter(MMDetModelAdapter, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modal = None
        self.modals = None

    def setup(self, stage = None):
        super().setup(stage)
        self.get_modals()

    def get_modals(self):
        if hasattr(self.dataset, 'Modals'):
            self.modal = self.dataset.Modal
            self.modals = self.dataset.Modals

    def predict_step(self, batch, *args, **kwargs):
        preds = self.model.simple_test(**batch)
        imgs = {m: batch[m]['img'].permute(0, 2, 3, 1).cpu().numpy() for m in self.modals}
        for i in range(len(preds)):
            pred = [bbox.cpu().numpy() for bbox in preds[i]]
            res = []
            for modal in [self.modal] + [m for m in self.modals if m != self.modal]:
                img = self.unnormarlize_img(imgs[modal][i], batch[modal]['img_metas'][i]['img_norm_cfg'])
                ann = {'gt_bboxes': batch[modal]['gt_bboxes'][i].cpu().numpy(), 'gt_labels': batch[modal]['gt_labels'][i].cpu().numpy()}
                res.append(imshow_gt_det_bboxes(img, ann, pred, class_names = self.class_names, show = False, **self.imshow_kwargs))
            res = np.concatenate(res, axis = 1)
            mmcv.imwrite(res, os.path.join(self.output_path, batch['img_metas'][i]['ori_filename']))
