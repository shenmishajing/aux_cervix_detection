import csv

import numpy as np
from mmcls.datasets import BaseDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class CataractDataSet(BaseDataset):
    CLASSES = [str(i) for i in range(7)]

    def load_annotations(self):
        data_infos = []
        for line in csv.reader(open(self.ann_file)):
            info = {'img_prefix': self.data_prefix, 'img_info': {'filename': line[0]}}
            for i, name in enumerate(['gt_label', 'center_label', 'center_turbidity', 'center_pervade_turbidity',
                                      'border_label', 'border_turbidity', 'border_pervade_turbidity']):
                info[name] = np.array(int(line[i + 1]), dtype = np.long)
            data_infos.append(info)
        return data_infos
