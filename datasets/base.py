import copy
from collections.abc import Mapping, Sequence

from pytorch_lightning.core.datamodule import LightningDataModule as _LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from utils.cli.yaml_with_merge import deep_update


class LightningDataModule(_LightningDataModule):
    SPLIT_NAMES = ['train', 'val', 'test', 'predict']

    def __init__(self, data_loader_config = None, split_name_map = None):
        super().__init__()
        if split_name_map is None:
            self.split_name_map = {}
        else:
            self.split_name_map = split_name_map
        for name in self.SPLIT_NAMES:
            self.split_name_map.setdefault(name, name if name != 'predict' else 'test')

        self.data_loader_config = {} if data_loader_config is None else data_loader_config

        if all([self.data_loader_config.get(name) is None for name in self.SPLIT_NAMES]):
            self.data_loader_config = {name: copy.deepcopy(self.data_loader_config) for name in self.SPLIT_NAMES}
        else:
            for main_name in self.SPLIT_NAMES:
                if self.data_loader_config.get(main_name) is not None:
                    for name in [n for n in self.SPLIT_NAMES if n != main_name]:
                        if self.data_loader_config.get(name) is None:
                            self.data_loader_config[name] = copy.deepcopy(self.data_loader_config[main_name])
                        else:
                            self.data_loader_config[name] = deep_update(copy.deepcopy(self.data_loader_config[main_name]),
                                                                        self.data_loader_config[name])
                    break

        self.datasets = {}

    def _setup_dataset(self, split_name):
        self.datasets[split_name] = self._build_data_set(self.split_name_map[split_name])

    def setup(self, stage = None):
        if self.trainer.overfit_batches > 0:
            split_names = ['train']
        elif stage is None:
            split_names = ['train', 'val', 'test', 'predict']
        elif stage == 'fit':
            split_names = ['train', 'val']
        else:
            split_names = [stage.lower()]

        for name in split_names:
            self._setup_dataset(name)

    def _dataloader(self, split_name, **kwargs):
        return self._build_data_loader(self.datasets[split_name], split = split_name, **kwargs)

    def train_dataloader(self):
        return self._dataloader('train', shuffle = True)

    def val_dataloader(self):
        return self._dataloader('val')

    def test_dataloader(self):
        return self._dataloader('test')

    def predict_dataloader(self):
        return self._dataloader('predict')

    def _build_data_set(self, split):
        raise NotImplementedError

    @staticmethod
    def collate(batch):
        raise NotImplementedError

    def _build_data_loader(self, dataset, shuffle = False, collate_fn = None, split = 'train'):
        def dataloader(ds, cl_fn) -> DataLoader:
            return DataLoader(ds, shuffle = shuffle and not isinstance(ds, IterableDataset), collate_fn = cl_fn,
                              **self.data_loader_config[split])

        if collate_fn is None:
            collate_fn = self.collate
        if isinstance(dataset, Mapping):
            return {key: self._build_data_loader(ds, shuffle = shuffle,
                                                 collate_fn = collate_fn[key] if isinstance(collate_fn, Mapping) else collate_fn,
                                                 split = split) for key, ds in dataset.items()}
        if isinstance(dataset, Sequence):
            return [self._build_data_loader(dataset[i], shuffle = shuffle,
                                            collate_fn = collate_fn[i] if isinstance(collate_fn, Sequence) else collate_fn, split = split)
                    for i in range(len(dataset))]
        return dataloader(dataset, cl_fn = collate_fn)
