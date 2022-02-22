import os

import torch

from .base import LightningDataModule
from .multi_modals import MultiModalsDataSet


class DualCervixDataSet(MultiModalsDataSet):
    CLASSES = ('hsil',)
    Modals = ['acid', 'iodine']


class DualCervixDataModule(LightningDataModule):
    def __init__(self,
                 ann_path,
                 pipeline,
                 modal = None,
                 data_root = '.',
                 img_prefix = '',
                 seg_prefix = '',
                 data_loader_config = None):
        super().__init__(data_loader_config)
        self.ann_path = ann_path
        self.pipeline = pipeline
        self.modal = modal
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix

    def _build_data_set(self, split):
        return DualCervixDataSet(ann_file = os.path.join(self.ann_path, split + '_{part}.json'),
                                 pipeline = self.pipeline,
                                 modal = self.modal,
                                 data_root = self.data_root,
                                 img_prefix = self.img_prefix,
                                 seg_prefix = self.seg_prefix)

    @staticmethod
    def collate_fn(batch):
        res = {}
        for part in batch[0]:
            res[part] = {}
            stack_keys = ['img', 'gt_segments_from_bboxes']
            for key in [k for k in stack_keys if k in batch[0][part]]:
                if isinstance(batch[0][part][key], torch.Tensor):
                    res[part][key] = torch.stack([x[part][key] for x in batch])
                else:
                    res[part][key] = [x[part][key] for x in batch]
            for key in [k for k in batch[0][part] if k not in stack_keys]:
                res[part][key] = [x[part][key] for x in batch]
        return res
