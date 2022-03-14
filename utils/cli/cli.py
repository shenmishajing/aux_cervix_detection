from types import MethodType
from typing import Any, Callable, Dict, Optional, Type, Union

from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, LightningArgumentParser, LightningCLI, SaveConfigCallback

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from utils.optim import get_configure_optimizers_method
from .trainer import Trainer, _Trainer
from .yaml_with_merge import ArgumentParser


class CLI(LightningCLI):
    def __init__(
            self,
            save_config_callback: Optional[Type[SaveConfigCallback]] = SaveAndLogConfigCallback,
            trainer_class: Union[Type[_Trainer], Callable[..., _Trainer]] = Trainer,
            *args, **kwargs
    ) -> None:
        super().__init__(save_config_callback = save_config_callback, trainer_class = trainer_class, *args, **kwargs)

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        return ArgumentParser(**kwargs)

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument(
            "--optimizer_config",
            type = Optional[Dict],
            default = None,
            help = "Configuration for the optimizers and lr schedulers."
        )

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_core_arguments_to_parser(parser)
        if self.datamodule_class is None and not len(DATAMODULE_REGISTRY):
            # this should not be required because the user might want to use the `LightningModule` dataloaders
            parser.add_lightning_class_args(self._datamodule_class, "data", subclass_mode = self.subclass_mode_data, required = False)

    def _add_configure_optimizers_method_to_model(self, subcommand: Optional[str]) -> None:
        super()._add_configure_optimizers_method_to_model(subcommand)
        optimizer_config = self._get(self.config_init, 'optimizer_config')
        if optimizer_config:
            self.model.configure_optimizers = MethodType(get_configure_optimizers_method(optimizer_config), self.model)
