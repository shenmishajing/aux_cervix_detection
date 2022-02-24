import os

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
        return DualCervixDataSet(ann_file = os.path.join(self.ann_path, split + '_{modal}.json'),
                                 pipeline = self.pipeline,
                                 modal = self.modal,
                                 data_root = self.data_root,
                                 img_prefix = self.img_prefix,
                                 seg_prefix = self.seg_prefix)
