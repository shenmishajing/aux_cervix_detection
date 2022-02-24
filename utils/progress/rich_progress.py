from typing import Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme


class RichDefaultThemeProgressBar(RichProgressBar):

    def __init__(
            self,
            refresh_rate_per_second: Optional[int] = 10,
            leave: Optional[bool] = False,
            show_version: Optional[bool] = True,
    ) -> None:
        super().__init__(refresh_rate_per_second = refresh_rate_per_second, leave = leave, theme = RichProgressBarTheme())
        self.show_version = show_version

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        if not self.show_version:
            # don't show the version number
            items.pop("v_num", None)
        return items

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f"Epoch {current_epoch}"
        if self.trainer.max_epochs is not None and self.trainer.max_epochs < 1000:
            train_description += f"/{self.trainer.max_epochs}"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            required_padding = len(self.validation_description) - len(train_description)
            train_description += " " * required_padding
        return train_description

    def on_train_epoch_start(self, trainer, pl_module):
        super(RichProgressBar, self).on_train_epoch_start(trainer, pl_module)

        train_description = self._get_train_description(trainer.current_epoch)
        if self.main_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.main_progress_bar_id is None:
            self.main_progress_bar_id = self._add_task(self.total_train_batches, train_description)
        elif self.progress is not None:
            self.progress.reset(self.main_progress_bar_id, total = self.total_train_batches,
                                description = train_description, visible = True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super(RichProgressBar, self).on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.sanity_checking:
            self._update(self.val_sanity_progress_bar_id)
        elif self.val_progress_bar_id is not None:
            self._update(self.val_progress_bar_id)
