import copy
import string
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from mmcv.parallel.data_container import DataContainer
from mmdet.datasets import build_dataset
from torch.utils.data.dataloader import default_collate

from ..base import LightningDataModule as _LightningDataModule


class MMDetDataSetAdapter(_LightningDataModule):
    def __init__(self,
                 type,
                 ann_file,
                 pipeline,
                 classes = None,
                 data_root = None,
                 img_prefix = '',
                 seg_prefix = None,
                 proposal_file = None,
                 test_mode = False,
                 filter_empty_gt = True,
                 file_client_args = dict(backend = 'disk'),
                 split_format_to = 'ann_file',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_format_to = split_format_to
        self.dataset_init_kwargs = {
            'type': type,
            'ann_file': ann_file,
            'pipeline': pipeline,
            'classes': classes,
            'data_root': data_root,
            'img_prefix': img_prefix,
            'seg_prefix': seg_prefix,
            'proposal_file': proposal_file,
            'test_mode': test_mode,
            'filter_empty_gt': filter_empty_gt,
            'file_client_args': file_client_args,
        }

    @staticmethod
    def collate(batch):
        """Puts each data field into a tensor/DataContainer with outer dimension
        batch size.

        Extend default_collate to add support for
        :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

        1. cpu_only = True, e.g., meta data
        2. cpu_only = False, stack = True, e.g., images tensors
        3. cpu_only = False, stack = False, e.g., gt bboxes
        """
        if not isinstance(batch, Sequence):
            raise TypeError(f'{batch.dtype} is not supported.')
        if isinstance(batch[0], DataContainer):
            if batch[0].stack:
                assert isinstance(batch[0].data, torch.Tensor)

                if batch[0].pad_dims is not None:
                    ndim = batch[0].dim()
                    assert ndim > batch[0].pad_dims
                    max_shape = [0 for _ in range(batch[0].pad_dims)]
                    for dim in range(1, batch[0].pad_dims + 1):
                        max_shape[dim - 1] = batch[0].size(-dim)
                    for sample in batch:
                        for dim in range(0, ndim - batch[0].pad_dims):
                            assert batch[0].size(dim) == sample.size(dim)
                        for dim in range(1, batch[0].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1], sample.size(-dim))
                    padded_samples = []
                    for sample in batch:
                        pad = [0 for _ in range(batch[0].pad_dims * 2)]
                        for dim in range(1, batch[0].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(F.pad(sample.data, pad, value = sample.padding_value))
                    return default_collate(padded_samples)
                elif batch[0].pad_dims is None:
                    return default_collate([sample.data for sample in batch])
                else:
                    raise ValueError('pad_dims should be either None or integers (1-3)')
            return [sample.data for sample in batch]
        elif isinstance(batch[0], Sequence):
            transposed = zip(*batch)
            return [MMDetDataSetAdapter.collate(samples) for samples in transposed]
        elif isinstance(batch[0], Mapping):
            return {key: MMDetDataSetAdapter.collate([d[key] for d in batch]) for key in batch[0]}
        else:
            return default_collate(batch)

    def _build_data_set(self, split):
        cfg = copy.deepcopy(self.dataset_init_kwargs)
        if self.split_format_to is None or self.split_format_to not in cfg:
            cfg['split'] = split
        else:
            cfg[self.split_format_to] = string.Template(cfg[self.split_format_to]).safe_substitute(split = split)
        return build_dataset(cfg)
