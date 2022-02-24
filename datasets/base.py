from collections.abc import Mapping, Sequence
from typing import Callable, Optional

from pytorch_lightning.core.datamodule import LightningDataModule as _LightningDataModule
from torch.utils.data import DataLoader, IterableDataset


class LightningDataModule(_LightningDataModule):
    def __init__(self, data_loader_config = None, split_name_map = None):
        super().__init__()
        for name in ['train', 'val', 'test', 'predict']:
            if split_name_map is not None:
                if name in split_name_map:
                    setattr(self, name + '_name', split_name_map[name])
                    continue
            if name == 'predict':
                setattr(self, name + '_name', 'test')
            else:
                setattr(self, name + '_name', name)
        if data_loader_config is None:
            self.data_loader_config = {}
        else:
            self.data_loader_config = data_loader_config

        self.train_dataset = self.val_dataset = self.test_dataset = self.predict_dataset = None

    def setup(self, stage = None):
        if self.trainer.overfit_batches > 0:
            self.train_dataset = self._build_data_set(self.train_name)
            return
        if stage in [None, 'fit']:
            self.train_dataset = self._build_data_set(self.train_name)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = self._build_data_set(self.val_name)
        if stage in [None, 'test']:
            self.test_dataset = self._build_data_set(self.test_name)
        if stage in [None, 'predict']:
            self.predict_dataset = self._build_data_set(self.predict_name)

    def train_dataloader(self):
        return self._build_data_loader(self.train_dataset, shuffle = True)

    def val_dataloader(self):
        return self._build_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self._build_data_loader(self.test_dataset)

    def predict_dataloader(self):
        return self._build_data_loader(self.predict_dataset)

    def _build_data_set(self, split):
        raise NotImplementedError

    @staticmethod
    def collate(batch):
        raise NotImplementedError

    def _build_data_loader(self, dataset, shuffle: Optional[bool] = False, collate_fn: Optional[Callable] = None):
        def dataloader(ds, cl_fn) -> DataLoader:
            return DataLoader(ds, shuffle = shuffle and not isinstance(ds, IterableDataset), collate_fn = cl_fn, **self.data_loader_config)

        if collate_fn is None:
            collate_fn = self.collate
        if isinstance(dataset, Mapping):
            return {key: dataloader(ds, cl_fn = collate_fn[key] if isinstance(collate_fn, Mapping) else collate_fn) for key, ds in
                    dataset.items()}
        if isinstance(dataset, Sequence):
            return [dataloader(dataset[i], cl_fn = collate_fn[i] if isinstance(collate_fn, Sequence) else collate_fn) for i in
                    range(len(dataset))]
        return dataloader(dataset, cl_fn = collate_fn)
