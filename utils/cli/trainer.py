from typing import Any, Dict, List
from typing import Optional

from pytorch_lightning import Trainer as _Trainer

from ..loop import KFoldLoop


class Trainer(_Trainer):
    def _configure_schedulers(
            self, schedulers: list, monitor: Optional[str], is_manual_optimization: bool
    ) -> List[Dict[str, Any]]:
        """Convert each scheduler into dict structure with relevant information."""
        return super()._configure_schedulers(schedulers, monitor, False)


class KFoldTrainer(Trainer):
    def __init__(self, num_folds: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_folds = num_folds

        internal_fit_loop = self.fit_loop
        self.fit_loop = KFoldLoop(self.num_folds)
        self.fit_loop.connect(internal_fit_loop)
