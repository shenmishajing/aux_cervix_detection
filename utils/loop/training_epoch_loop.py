from typing import List, Optional

from pytorch_lightning.loops import TrainingEpochLoop as _TrainingEpochLoop
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TrainingEpochLoop(_TrainingEpochLoop):
    def _update_learning_rates(
            self, interval: str, update_plateau_schedulers: bool, opt_indices: Optional[List[int]] = None
    ) -> None:
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            update_plateau_schedulers: control whether ``ReduceLROnPlateau`` or non-plateau schedulers get updated.
                This is used so non-plateau schedulers can be updated before running validation. Checkpoints are
                commonly saved during validation, however, on-plateau schedulers might monitor a validation metric
                so they have to be updated separately.
            opt_indices: indices of the optimizers to update.
        """
        if not self.trainer.lr_schedulers or not self.trainer.lightning_module.automatic_optimization and \
                not self.trainer.lightning_module.automatic_lr_schedule:
            return

        if opt_indices is None:
            opt_indices = []

        for lr_scheduler in self.trainer.lr_schedulers:
            if isinstance(lr_scheduler["opt_idx"], int) and lr_scheduler["opt_idx"] not in opt_indices:
                continue

            if update_plateau_schedulers ^ lr_scheduler["reduce_on_plateau"]:
                continue

            current_idx = self.batch_idx if interval == "step" else self.trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler["interval"] == interval and current_idx % lr_scheduler["frequency"] == 0:
                monitor_val = None
                if lr_scheduler["reduce_on_plateau"]:
                    # If instance of ReduceLROnPlateau, we need a monitor
                    monitor_key = lr_scheduler["monitor"]
                    monitor_val = self._get_monitor_value(monitor_key)
                    if monitor_val is None:
                        if lr_scheduler.get("strict", True):
                            avail_metrics = list(self.trainer.callback_metrics)
                            raise MisconfigurationException(
                                f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                                f" which is not available. Available metrics are: {avail_metrics}."
                                " Condition can be set using `monitor` key in lr scheduler dict"
                            )
                        rank_zero_warn(
                            f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                            " which is not available but strict is set to `False`."
                            " Skipping learning rate update.",
                            RuntimeWarning,
                        )
                        continue

                self.scheduler_progress.increment_ready()

                # update LR
                if lr_scheduler["reduce_on_plateau"]:
                    lr_scheduler["scheduler"].step(monitor_val)
                else:
                    lr_scheduler["scheduler"].step()

                self.scheduler_progress.increment_completed()
