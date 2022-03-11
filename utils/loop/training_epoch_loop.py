from pytorch_lightning.loops import TrainingEpochLoop as _TrainingEpochLoop


class TrainingEpochLoop(_TrainingEpochLoop):
    def _update_learning_rates(self, *args, **kwargs) -> None:
        automatic_optimization = self.trainer.lightning_module.automatic_optimization
        self.trainer.lightning_module.automatic_optimization |= self.trainer.lightning_module.automatic_lr_schedule
        super(TrainingEpochLoop, self)._update_learning_rates(*args, **kwargs)
        self.trainer.lightning_module.automatic_optimization = automatic_optimization
