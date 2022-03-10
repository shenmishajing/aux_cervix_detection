from typing import Any, Callable, Optional, Type, Union

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY, DATAMODULE_REGISTRY, LightningArgumentParser, LightningCLI, MODEL_REGISTRY, \
    SaveConfigCallback, instantiate_class

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from .yaml_with_merge import ArgumentParser


class CLI(LightningCLI):
    def __init__(
            self,
            save_config_callback: Optional[Type[SaveConfigCallback]] = SaveAndLogConfigCallback,
            trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = None,
            subclass_mode_trainer: bool = False,
            *args, **kwargs
    ) -> None:
        if trainer_class is not None:
            kwargs['trainer_class'] = trainer_class
        # used to differentiate between the original value and the processed value
        self._trainer_class = trainer_class or Trainer
        self.subclass_mode_trainer = (trainer_class is None) or subclass_mode_trainer
        super().__init__(save_config_callback = save_config_callback, *args, **kwargs)

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        return ArgumentParser(**kwargs)

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        parser.add_lightning_class_args(self.trainer_class, "trainer", subclass_mode = self.subclass_mode_trainer)
        parser.set_choices("trainer." + ("init_args." if self.subclass_mode_trainer else "") + "callbacks", CALLBACK_REGISTRY.classes,
                           is_list = True)
        trainer_defaults = {"trainer." + ("init_args." if self.subclass_mode_trainer else "") + k: v for k, v in
                            self.trainer_defaults.items() if k != "callbacks"}
        parser.set_defaults(trainer_defaults)

        parser.add_lightning_class_args(self._model_class, "model", subclass_mode = self.subclass_mode_model)
        if self.model_class is None and len(MODEL_REGISTRY):
            # did not pass a model and there are models registered
            parser.set_choices("model", MODEL_REGISTRY.classes)

        if self.datamodule_class is not None:
            parser.add_lightning_class_args(self._datamodule_class, "data", subclass_mode = self.subclass_mode_data)
        else:
            # this should not be required because the user might want to use the `LightningModule` dataloaders
            parser.add_lightning_class_args(
                self._datamodule_class, "data", subclass_mode = self.subclass_mode_data, required = False
            )
            if len(DATAMODULE_REGISTRY):
                parser.set_choices("data", DATAMODULE_REGISTRY.classes)

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self._get(self.config_init, "trainer")

        if "callbacks" in self.trainer_defaults:
            if not isinstance(self.trainer_defaults["callbacks"], list):
                callbacks = [self.trainer_defaults["callbacks"]]
            else:
                callbacks = self.trainer_defaults["callbacks"]
            for c in callbacks:
                self.trainer.callbacks.append(instantiate_class(None, c))
        if self.save_config_callback and not self.trainer.fast_dev_run:
            config_callback = self.save_config_callback(
                self._parser(self.subcommand),
                self.config.get(str(self.subcommand), self.config),
                self.save_config_filename,
                overwrite = self.save_config_overwrite,
                multifile = self.save_config_multifile,
            )
            self.trainer.callbacks.append(config_callback)
