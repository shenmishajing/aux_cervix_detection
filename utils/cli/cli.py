from typing import Any, Callable, Optional, Type, Union

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, LightningArgumentParser, LightningCLI, SaveConfigCallback

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from .trainer import Trainer as _Trainer
from .yaml_with_merge import ArgumentParser

DATAMODULE_REGISTRY(object)


class CLI(LightningCLI):
    def __init__(
            self,
            save_config_callback: Optional[Type[SaveConfigCallback]] = SaveAndLogConfigCallback,
            trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = _Trainer,
            *args, **kwargs
    ) -> None:
        super().__init__(save_config_callback = save_config_callback, trainer_class = trainer_class, *args, **kwargs)

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        return ArgumentParser(**kwargs)
