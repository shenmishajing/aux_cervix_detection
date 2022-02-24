from mmdet.datasets import DATASETS

from .multi_modals import MultiModalsDataSet


@DATASETS.register_module()
class DualCervixDataSet(MultiModalsDataSet):
    CLASSES = ('hsil',)
    Modals = ['acid', 'iodine']
