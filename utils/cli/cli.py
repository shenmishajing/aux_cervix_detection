import os
from typing import Any, Callable, Optional, Type, Union

from jsonargparse.jsonnet import ActionJsonnet
from jsonargparse.loaders_dumpers import get_loader_exceptions, load_value
from jsonargparse.namespace import Namespace
from jsonargparse.optionals import get_config_read_mode, import_jsonnet
from jsonargparse.util import Path, change_to_path_dir
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, LightningArgumentParser, LightningCLI, SaveConfigCallback

from utils.callbacks.save_and_log_config_callback import SaveAndLogConfigCallback
from .trainer import Trainer as _Trainer
from .yaml import parse_str

DATAMODULE_REGISTRY(object)


class ArgumentParser(LightningArgumentParser):
    def _load_config_parser_mode(
            self,
            cfg_str: str,
            cfg_path: str = '',
            ext_vars: Optional[dict] = None,
    ) -> Namespace:
        """Loads a configuration string (yaml or jsonnet) into a namespace.

        Args:
            cfg_str: The configuration content.
            cfg_path: Optional path to original config path, just for error printing.
            ext_vars: Optional external variables used for parsing jsonnet.

        Raises:
            TypeError: If there is an invalid value according to the parser.
        """
        if self.parser_mode == 'jsonnet':
            ext_vars, ext_codes = ActionJsonnet.split_ext_vars(ext_vars)
            _jsonnet = import_jsonnet('_load_config_parser_mode')
            cfg_str = _jsonnet.evaluate_snippet(cfg_path, cfg_str, ext_vars = ext_vars, ext_codes = ext_codes)
        try:
            if self.parser_mode == 'jsonnet':
                cfg_dict = load_value(cfg_str)
            else:
                cfg_dict = parse_str(cfg_str, cfg_path = cfg_path)
        except get_loader_exceptions() as ex:
            raise TypeError(f'Problems parsing config :: {ex}') from ex

        cfg = self._apply_actions(cfg_dict)

        return cfg

    def parse_path(
            self,
            cfg_path: str,
            ext_vars: Optional[dict] = None,
            env: Optional[bool] = None,
            defaults: bool = True,
            with_meta: Optional[bool] = None,
            _skip_check: bool = False,
            _fail_no_subcommand: bool = True,
    ) -> Namespace:
        """Parses a configuration file (yaml or jsonnet) given its path.

        Args:
            cfg_path: Path to the configuration file to parse.
            ext_vars: Optional external variables used for parsing jsonnet.
            env: Whether to merge with the parsed environment, None to use parser's default.
            defaults: Whether to merge with the parser's defaults.
            with_meta: Whether to include metadata in config object, None to use parser's default.

        Returns:
            A config object with all parsed values.

        Raises:
            ParserError: If there is a parsing error and error_handler=None.
        """
        fpath = Path(cfg_path, mode = get_config_read_mode())
        with change_to_path_dir(fpath):
            cfg_str = fpath.get_content()
            parsed_cfg = self.parse_string(cfg_str,
                                           os.path.basename(cfg_path),
                                           ext_vars,
                                           env,
                                           defaults,
                                           with_meta = with_meta,
                                           _skip_check = _skip_check,
                                           _fail_no_subcommand = _fail_no_subcommand)

        self._logger.info(f'Parsed {self.parser_mode} from path: {cfg_path}')

        return parsed_cfg


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
