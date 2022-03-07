from mmdet.datasets import CocoDataset, DATASETS


@DATASETS.register_module()
class TCTDataSet(CocoDataset):
    CLASSES = ('ASCH', 'ASCUS', 'HSIL', 'LSIL', 'SQCA')
