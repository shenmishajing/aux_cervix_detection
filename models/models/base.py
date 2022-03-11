import copy
from abc import ABC
from typing import Any, Mapping, Sequence, Union

import torch
from mmcv.runner import BaseModule
from pytorch_lightning import LightningModule as _LightningModule

from utils import optim


class LightningModule(_LightningModule, BaseModule, ABC):
    def __init__(self,
                 normalize_config = None,
                 optimizer_config = None,
                 loss_config: Mapping[str, Union[torch.nn.Module, Mapping[str, Union[torch.nn.Module, int, float]]]] = None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_lr_schedule = True
        self.normalize_config = normalize_config
        self.optimizer_config = optimizer_config
        self.lr = None
        self.batch_size = None

        if loss_config is not None:
            self._parse_loss_config(loss_config)

    @staticmethod
    def add_prefix(log_dict, prefix = 'train/'):
        return {f'{prefix}{k}': v for k, v in log_dict.items()}

    def log(self, *args, batch_size = None, **kwargs):
        if batch_size is None and hasattr(self, 'batch_size') and self.batch_size is not None:
            batch_size = self.batch_size
        super().log(*args, batch_size = batch_size, **kwargs)

    def _parse_loss_config(self, loss_config):
        for key, value in loss_config.items():
            if not isinstance(value, Mapping):
                loss_config[key] = {'module': value, 'weight': 1}
            setattr(self, 'loss_' + key, loss_config[key]['module'])
        self.loss_weight = {k: v.get('weight', 1) for k, v in loss_config.items()}

    def _loss_step(self, batch, res, prefix = 'train'):
        raise NotImplementedError

    def loss_step(self, batch, res, prefix = 'train', use_loss_weight = True, loss_use_loss_weight = True, detach = None):
        loss = self._loss_step(batch, res, prefix)
        # multi loss weights
        if use_loss_weight:
            loss = {k: v * (1 if k not in self.loss_weight else self.loss_weight[k]) for k, v in loss.items()}
        # calculate loss
        if not use_loss_weight and loss_use_loss_weight:
            total_loss = [v * (1 if k not in self.loss_weight else self.loss_weight[k]) for k, v in loss.items()]
        else:
            total_loss = [v for k, v in loss.items()]
        loss['loss'] = torch.sum(torch.stack(total_loss))
        # add prefix
        if detach is None:
            detach = prefix != 'train'
        loss = {(f'{prefix}/' if prefix is not None else '') + ('loss_' if 'loss' not in k else '') + k: (v.detach() if detach else v) for
                k, v in loss.items()}
        return loss

    def on_fit_start(self):
        self.init_weights()

    def _dump_init_info(self, logger_name):
        pass

    def training_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'train')
        self.log_dict(loss)
        return loss['train/loss']

    def validation_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'val')
        self.log_dict(loss)
        return loss

    def test_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss = self.loss_step(batch, res, 'test')
        self.log_dict(loss)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers for model.
        """
        optimizer_config = self._parse_optimizer_config(self.optimizer_config)
        # construct optimizer
        optimizer_config['optimizer'] = self._construct_optimizers(optimizer_config['optimizer'])
        # construct lr_scheduler
        if 'lr_scheduler' in optimizer_config:
            optimizer_config['lr_scheduler'] = self._construct_lr_schedulers(optimizer_config['lr_scheduler'],
                                                                             optimizer_config['optimizer'])
            return optimizer_config['optimizer'], optimizer_config['lr_scheduler']
        return optimizer_config['optimizer']

    def configure_optimizer_parameters(self):
        return None

    def _parse_optimizer_config(self, optimizer_config):
        optimizer_config = copy.deepcopy(optimizer_config)
        assert isinstance(optimizer_config, dict), 'optimizer_config should be a dict'

        if 'optimizer' not in optimizer_config:
            optimizer_config = {'optimizer': optimizer_config}
        if not isinstance(optimizer_config['optimizer'], Sequence):
            parameters = self.configure_optimizer_parameters()
            if parameters is not None:
                optimizer_config['optimizer'] = [copy.deepcopy(optimizer_config['optimizer']) for _ in parameters]
            else:
                optimizer_config['optimizer'] = [optimizer_config['optimizer']]

        if 'lr_scheduler' in optimizer_config:
            if not isinstance(optimizer_config['lr_scheduler'], Sequence):
                optimizer_config['lr_scheduler'] = [copy.deepcopy(optimizer_config['lr_scheduler']) for _ in optimizer_config['optimizer']]
        return optimizer_config

    def _construct_optimizers(self, optimizers):
        """
        Constructs all optimizers.

        Args:
            optimizers: list of dictionary containing optimizer configuration.
        """
        optimizer_parameters = self.configure_optimizer_parameters()
        if optimizer_parameters is None:
            optimizer_parameters = [None] * len(optimizers)
        assert isinstance(optimizer_parameters, list), 'optimizer_parameters should be None or list'
        if len(optimizer_parameters) < len(optimizers):
            optimizer_parameters += [None] * (len(optimizers) - len(optimizer_parameters))

        for i in range(len(optimizers)):
            optimizers[i] = self._construct_optimizer(optimizers[i], optimizer_parameters[i], set_lr = i == 0)

        return optimizers

    def _construct_optimizer(self, optimizer, params = None, set_lr = False):
        """
        Constructs the optimizer.

        Args:
            optimizer: dictionary containing optimizer configuration.
        """
        optimizer_type = optimizer.pop('type')
        if self.lr is not None and set_lr:
            optimizer['lr'] = self.lr
        optimizer = optim.__dict__[optimizer_type](self.parameters() if params is None else params, **optimizer)
        if set_lr and self.lr is None:
            self.lr = optimizer.param_groups[0]['lr']
        return optimizer

    def _construct_lr_schedulers(self, lr_schedulers, optimizers):
        """
        Constructs all lr_schedulers.

        Args:
            lr_schedulers: list of dictionary containing lr_scheduler configuration.
            optimizers: list of optimizers constructed.
        """
        constructed_lr_schedulers = []
        warmup_lr_schedulers = []
        opt_idx = -1
        # construct lr_scheduler
        for lr_scheduler in lr_schedulers:
            # select optimizer
            if len(optimizers) == 1:
                opt_idx = 0
            else:
                if 'opt_idx' in lr_scheduler:
                    opt_idx = lr_scheduler['opt_idx']
                else:
                    opt_idx += 1
            optimizer = optimizers[opt_idx]

            # construct lr_scheduler
            if 'scheduler' not in lr_scheduler:
                lr_scheduler = {'scheduler': lr_scheduler}
            lr_scheduler['scheduler'] = self._construct_lr_scheduler(optimizer, lr_scheduler['scheduler'])
            lr_scheduler['opt_idx'] = opt_idx
            constructed_lr_schedulers.append(lr_scheduler)

            # construct warmup_lr_scheduler
            if 'warmup_config' in lr_scheduler:
                warmup_config = lr_scheduler.pop('warmup_config')
                if 'scheduler' not in warmup_config:
                    warmup_config = {'scheduler': warmup_config}
                warmup_config['scheduler']['type'] = 'WarmupScheduler'
                warmup_config['scheduler'] = self._construct_lr_scheduler(optimizer, warmup_config['scheduler'])
                warmup_config.update({'interval': 'step', 'opt_idx': opt_idx})
                warmup_lr_schedulers.append(warmup_config)
        return constructed_lr_schedulers + warmup_lr_schedulers

    @staticmethod
    def _construct_lr_scheduler(optimizer, lr_scheduler):
        """
        Constructs the lr_scheduler.

        Args:
            optimizer: the optimizer used to construct the lr_scheduler.
            lr_scheduler: dictionary containing lr_scheduler configuration.
        """
        lr_scheduler_type = lr_scheduler.pop('type')
        lr_scheduler = optim.__dict__[lr_scheduler_type](optimizer, **lr_scheduler)
        return lr_scheduler
