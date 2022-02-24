import string

import numpy as np
from mmdet.datasets import CocoDataset, DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.pipelines import Compose

from .mmdataset_adapter import MMDetDataSetAdapter


@DATASETS.register_module()
class MultiModalsDataSet(CocoDataset):
    Modals = []

    @classmethod
    def set_main_modal(cls, modal = None):
        if modal:
            cls.Modal = modal
        else:
            cls.Modal = cls.Modals[0]

    def __init__(self, ann_file, pipeline, modal = None, *args, **kwargs):
        self.set_main_modal(modal)
        super().__init__(ann_file = ann_file, pipeline = [], *args, **kwargs)
        self.modal = modal
        # processing pipeline
        if not isinstance(pipeline, dict):
            pipeline = {modal: pipeline for modal in self.Modals}
        self.pipeline = {modal: Compose(pipeline[modal]) for modal in self.Modals}

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            dict[str, list[dict]]: Annotation info from COCO api.
        """

        self._coco = {modal: COCO(string.Template(ann_file).substitute(modal = modal)) for modal in self.Modals}
        self.coco = self._coco[self.Modal]
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names = self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = {modal: list() for modal in self.Modals}
        for i in self.img_ids:
            data_info = {}
            for modal in self.Modals:
                info = self._coco[modal].load_imgs([i])[0]
                info['filename'] = info['file_name']
                data_info[modal] = info
                ann_ids = self._coco[modal].get_ann_ids(img_ids = [i])
                total_ann_ids[modal].extend(ann_ids)
            data_infos.append(data_info)
        for modal in self.Modals:
            assert len(set(total_ann_ids[modal])) == len(total_ann_ids[modal]), \
                f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx, modal = None):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx][self.Modal]['id']
        if modal is None:
            modal = self.Modal
        ann_ids = self._coco[modal].get_ann_ids(img_ids = [img_id])
        ann_info = self._coco[modal].load_anns(ann_ids)
        ann_info = self._parse_ann_info(self.data_infos[idx][modal], ann_info)
        return ann_info

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx][self.Modal]['id']
        ann_ids = {modal: self._coco[modal].get_ann_ids(img_ids = [img_id]) for modal in self.Modals}
        ann_info = {modal: [ann['category_id'] for ann in self._coco[modal].load_anns(ann_ids[modal])] for modal in self.Modals}
        return ann_info

    def _filter_imgs(self, min_size = 32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            img_info = img_info[self.Modal]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype = np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i][self.Modal]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def prepare_img_on_modal(self, idx, modal, training = True):
        img_info = self.data_infos[idx][modal]
        if training:
            ann_info = self.get_ann_info(idx, modal)
            results = dict(img_info = img_info, ann_info = ann_info)
        else:
            results = dict(img_info = img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline[modal](results)

    def prepare_img(self, idx, training = True):
        """Get data and optional annotations after pipeline.

        Args:
            idx (int): Index of data.
            training (bool): Get training data or test data. Default: True.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        if self.modal is None:
            res = {}
            random_state = np.random.get_state()
            for modal in self.Modals:
                np.random.set_state(random_state)
                res[modal] = self.prepare_img_on_modal(idx, modal, training)
        else:
            res = self.prepare_img_on_modal(idx, self.modal, training)
        return res

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        return self.prepare_img(idx, training = True)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        return self.prepare_img(idx, training = False)


class MMDetMultiModelDataSetAdapter(MMDetDataSetAdapter):
    def __init__(self, modal = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_init_kwargs['modal'] = modal
