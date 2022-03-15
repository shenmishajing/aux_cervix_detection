from typing import Dict, Optional

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class WandbNamedLogger(WandbLogger):

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment if the experiment exists else the name given to the constructor.
        """
        # don't create an experiment if we don't have one
        return self._experiment.name if self._experiment else self._name

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = self._add_prefix(metrics)
        if step is not None:
            self.experiment.log({**metrics, "global_step": step})
        else:
            self.experiment.log(metrics)
