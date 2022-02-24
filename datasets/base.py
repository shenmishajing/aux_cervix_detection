from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from mmcv.parallel.data_container import DataContainer
from pytorch_lightning.core.datamodule import LightningDataModule as _LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import default_collate


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
        return [collate(samples) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return default_collate(batch)


class LightningDataModule(_LightningDataModule, ABC):
    def __init__(self, data_loader_config: Optional[Mapping[str, Any]] = None):
        super().__init__()

        if data_loader_config is None:
            self.data_loader_config = {}
        else:
            self.data_loader_config = data_loader_config

        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer.overfit_batches > 0:
            self.train_dataset = self._build_data_set('train')
            return
        if stage in [None, 'fit']:
            self.train_dataset = self._build_data_set('train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = self._build_data_set('val')
        if stage in [None, 'test', 'predict']:
            self.test_dataset = self._build_data_set('test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._build_data_loader(self.train_dataset, shuffle = True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._build_data_loader(self.val_dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._build_data_loader(self.test_dataset)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._build_data_loader(self.test_dataset)

    def _build_data_set(self, split):
        raise NotImplementedError

    def _build_data_loader(self, dataset, shuffle: Optional[bool] = False, collate_fn: Optional[Callable] = None) -> TRAIN_DATALOADERS:
        def dataloader(ds, cl_fn) -> DataLoader:
            return DataLoader(ds, shuffle = shuffle and not isinstance(ds, IterableDataset), collate_fn = cl_fn, **self.data_loader_config)

        if collate_fn is None:
            if hasattr(self, 'collate_fn') and self.collate_fn is not None:
                collate_fn = self.collate_fn
            else:
                collate_fn = collate
        if isinstance(dataset, Mapping):
            return {key: dataloader(ds, cl_fn = collate_fn[key] if isinstance(collate_fn, Mapping) else collate_fn) for key, ds in
                    dataset.items()}
        if isinstance(dataset, Sequence):
            return [dataloader(dataset[i], cl_fn = collate_fn[i] if isinstance(collate_fn, Sequence) else collate_fn) for i in
                    range(len(dataset))]
        return dataloader(dataset, cl_fn = collate_fn)
