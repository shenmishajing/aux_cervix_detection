import csv

import numpy as np
from mmcls.datasets import BaseDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class ISIC2017SkinDataSet(BaseDataset):
    CLASSES = ['False', 'True']

    def __init__(self,
                 is_melanoma = True,
                 *args, **kwargs):
        self.is_melanoma = is_melanoma
        super().__init__(*args, **kwargs)

    def load_annotations(self):
        data_infos = []
        for line in csv.reader(open(self.ann_file)):
            info = {
                'img_prefix': self.data_prefix,
                'img_info': {'filename': f'{line[0]}.jpg'},
                'gt_label': np.array(line[1 if self.is_melanoma else 2], dtype = np.int64)
            }
            data_infos.append(info)
        return data_infos
