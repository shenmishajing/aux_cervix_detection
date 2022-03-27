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
            info = {
                'img_prefix': self.data_prefix,
                'img_info': {'filename': line[0]},
                'gt_label': np.array(line[1], dtype = np.int64),
                'gt_min_label': np.array(line[2], dtype = np.int64),
                'gt_max_label': np.array(line[3], dtype = np.int64)
            }
            if len(line) > 4:
                info['label'] = np.array(line[4:], dtype = np.int64)
            data_infos.append(info)
        return data_infos
