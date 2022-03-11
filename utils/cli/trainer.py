from typing import Any, Dict, List
from typing import Optional

from pytorch_lightning import Trainer as _Trainer
from pytorch_lightning.trainer.connectors.env_vars_connector import _defaults_from_env_vars

from ..loop import KFoldLoop, TrainingEpochLoop


class Trainer(_Trainer):
    @_defaults_from_env_vars
    def __init__(
            self,
            num_folds: Optional[int] = None,
            max_steps: int = -1,
            min_steps: Optional[int] = None,
            *args, **kwargs
    ):
        super().__init__(max_steps = max_steps, min_steps = min_steps, *args, **kwargs)

        # add automatic_lr_schedule support
        training_epoch_loop = TrainingEpochLoop(min_steps, max_steps)
        training_epoch_loop.connect(batch_loop = self.fit_loop.epoch_loop.batch_loop, val_loop = self.fit_loop.epoch_loop.val_loop)
        self.fit_loop.connect(epoch_loop = training_epoch_loop)

        # add kfold cross validation support
        self.num_folds = num_folds
        if self.num_folds is not None and self.num_folds > 1:
            self.fit_loop = KFoldLoop(self.num_folds, self.fit_loop)

    def _configure_schedulers(
            self, schedulers: list, monitor: Optional[str], is_manual_optimization: bool
    ) -> List[Dict[str, Any]]:
        """Convert each scheduler into dict structure with relevant information."""
        return super()._configure_schedulers(schedulers, monitor, False)
