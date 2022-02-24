# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import CutOut as _CutOut
from mmdet.datasets.pipelines import DefaultFormatBundle as _DefaultFormatBundle
from mmdet.datasets.pipelines import MixUp as _MixUp
from mmdet.datasets.pipelines import Mosaic as _Mosaic
from mmdet.datasets.pipelines import Resize as _Resize
from mmdet.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module()
class GenSegmentationFromBBox:
    """Generate segmentation from bbox.

    Added key is "gt_segments_from_bboxes".

    Args:
        num_classes (int): number of classes. Generate instance segmentation,
            if is None, else semantic segmentation. Default : None.
    """

    def __init__(self, num_classes = None):
        self.num_classes = num_classes

    def __call__(self, results):
        """Call function to generate segmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Segmentation generated results, 'gt_segments_from_bboxes' key
                is added into result dict.
        """
        img_shape = results['img_shape'][:2]
        if self.num_classes is None:
            kernels = [self.__gen_gaussian_kernel(bbox, img_shape) for bbox in results['gt_bboxes']]
        else:
            kernels = np.zeros((self.num_classes, *img_shape), dtype = np.float32)
            for bbox, label in zip(results['gt_bboxes'], results['gt_labels']):
                kernels[label - 1, :, :] += self.__gen_gaussian_kernel(bbox, img_shape)

        results['gt_segments_from_bboxes'] = kernels
        results['seg_fields'].append('gt_segments_from_bboxes')
        return results

    @staticmethod
    def __gen_gaussian_kernel(bbox, img_shape):
        mu = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
        sigma = [(bbox[3] - bbox[1]) / 3, (bbox[2] - bbox[0]) / 3]
        h, w = [np.exp(-np.power((np.arange(1., sh + 1) - m) / s, 2) / 2) for sh, m, s in zip(img_shape, mu, sigma)]
        kernel = h[:, np.newaxis] * w[np.newaxis, :]
        kernel = 9 * sigma[0] * sigma[1] * kernel / np.sum(kernel)
        return kernel.astype(np.float32)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(num_classes={self.num_classes})'
        return repr_str


@PIPELINES.register_module(force = True)
class DefaultFormatBundle(_DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    - gt_segments_from_bboxes (1)to tensor, (2)to DataContainer (stack=True)
    """

    def __call__(self, results):
        results = super().__call__(results)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(to_tensor(results['gt_semantic_seg']), stack = True)
        return results


@PIPELINES.register_module(force = True)
class Resize(_Resize):
    def __init__(self, img_scale = None, *args, **kwargs):
        if isinstance(img_scale, list):
            if not mmcv.is_list_of(img_scale, tuple):
                img_scale = [tuple(img_scale)]
        super().__init__(img_scale = img_scale, *args, **kwargs)


@PIPELINES.register_module(force = True)
class CutOut(_CutOut):
    def __init__(self, n_holes, *args, **kwargs):
        if isinstance(n_holes, list):
            n_holes = tuple(n_holes)
        super().__init__(n_holes = n_holes, *args, **kwargs)


@PIPELINES.register_module(force = True)
class Mosaic(_Mosaic):
    def __init__(self, img_scale = (640, 640), *args, **kwargs):
        if isinstance(img_scale, list):
            img_scale = tuple(img_scale)
        super().__init__(img_scale = img_scale, *args, **kwargs)


@PIPELINES.register_module(force = True)
class MixUp(_MixUp):
    def __init__(self, img_scale = (640, 640), *args, **kwargs):
        if isinstance(img_scale, list):
            img_scale = tuple(img_scale)
        super().__init__(img_scale = img_scale, *args, **kwargs)
