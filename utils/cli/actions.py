import os
import copy
from typing import Dict, List, Optional
from jsonargparse.util import change_to_path_dir
from jsonargparse import ActionConfigFile, get_config_read_mode, Path
from jsonargparse.actions import _ActionSubCommands
from jsonargparse.loaders_dumpers import load_value, get_loader_exceptions


def deep_update(source, override):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    if isinstance(source, Dict) and isinstance(override, Dict):
        if '__delete__' in override:
            delete_keys = override.pop('__delete__')
            if isinstance(delete_keys, str):
                delete_keys = [delete_keys]

            if isinstance(delete_keys, list):
                for k in delete_keys:
                    if k in source:
                        source.pop(k)
            elif delete_keys:
                return override
        for key, value in override.items():
            if isinstance(value, Dict) and key in source:
                source[key] = deep_update(source[key], value)
            else:
                source[key] = override[key]
        return source
    elif isinstance(source, List) and isinstance(override, Dict):
        if 'change_item' in override:
            change_item = override.pop('change_item')
            for index, v in change_item:
                source[index] = deep_update(source[index], v)

        if '__delete__' in override:
            delete_keys = override.pop('__delete__')
            if isinstance(delete_keys, int):
                delete_keys = [delete_keys]

            if isinstance(delete_keys, list):
                delete_keys = list({int(d) for d in delete_keys})
                delete_keys.sort(reverse = True)
                for k in delete_keys:
                    source.pop(k)
            elif delete_keys:
                return override
        if 'pre_item' in override:
            source = (override['pre_item'] if isinstance(override['pre_item'], list) else [override['pre_item']]) + source
        if 'post_item' in override:
            source = source + (override['post_item'] if isinstance(override['post_item'], list) else [override['post_item']])
        return source
    return override


def get_cfg_from_str(cfg_str, **kwargs):
    cfg = load_value(cfg_str)
    return cfg


def get_cfg_from_path(cfg_path, **kwargs):
    fpath = Path(cfg_path, mode = get_config_read_mode())
    with change_to_path_dir(fpath):
        cfg_str = fpath.get_content()
        parsed_cfg = get_cfg_from_str(cfg_str, **kwargs)
    return parsed_cfg


def parse_config(cfg_file, cfg_path = None, **kwargs):
    if '__base__' in cfg_file:
        sub_cfg_paths = cfg_file.pop('__base__')
        if sub_cfg_paths is not None:
            if not isinstance(sub_cfg_paths, list):
                sub_cfg_paths = [sub_cfg_paths]
            sub_cfg_paths = [sub_cfg_path if isinstance(sub_cfg_path, list) else [sub_cfg_path, ''] for sub_cfg_path in sub_cfg_paths]
            if cfg_path is not None:
                sub_cfg_paths = [[os.path.normpath(os.path.join(os.path.dirname(cfg_path), sub_cfg_path[0])) if not os.path.isabs(
                    sub_cfg_path[0]) else sub_cfg_path[0], sub_cfg_path[1]] for sub_cfg_path in sub_cfg_paths]
            sub_cfg_file = {}
            for sub_cfg_path in sub_cfg_paths:
                cur_cfg_file = _parse_path(sub_cfg_path[0], **kwargs)
                for key in sub_cfg_path[1].split('.'):
                    if key:
                        cur_cfg_file = cur_cfg_file[key]
                sub_cfg_file = deep_update(sub_cfg_file, cur_cfg_file)
            cfg_file = deep_update(sub_cfg_file, cfg_file)
    if '__import__' in cfg_file:
        cfg_file.pop('__import__')

    for k, v in cfg_file.items():
        if isinstance(v, dict):
            cfg_file[k] = parse_config(v, cfg_path, **kwargs)
    return cfg_file


def _parse_path(cfg_path, seen_cfg = None, **kwargs):
    abs_cfg_path = os.path.abspath(cfg_path)
    if seen_cfg is None:
        seen_cfg = {}
    elif abs_cfg_path in seen_cfg:
        if seen_cfg[abs_cfg_path] is None:
            raise RuntimeError('Circular reference detected in config file')
        else:
            return copy.deepcopy(seen_cfg[abs_cfg_path])

    cfg_file = get_cfg_from_path(cfg_path, **kwargs)
    seen_cfg[abs_cfg_path] = None
    cfg_file = parse_config(cfg_file, cfg_path = cfg_path, seen_cfg = seen_cfg, **kwargs)
    seen_cfg[abs_cfg_path] = cfg_file
    return cfg_file


def parse_path(parser, cfg_path, **kwargs):
    return parser._apply_actions(_parse_path(cfg_path, parser = parser, **kwargs))


def _parse_string(cfg_string, **kwargs):
    return parse_config(get_cfg_from_str(cfg_string, **kwargs), **kwargs)


def parse_string(parser, cfg_string, **kwargs):
    return parser._apply_actions(_parse_string(cfg_string, parser = parser, **kwargs))


class LightningActionConfigFile(ActionConfigFile):
    @staticmethod
    def apply_config(parser, cfg, dest, value) -> None:
        with _ActionSubCommands.not_single_subcommand():
            if dest not in cfg:
                cfg[dest] = []
            kwargs = {'env': False, 'defaults': False, '_skip_check': True, '_fail_no_subcommand': False}
            try:
                cfg_path: Optional[Path] = Path(value, mode = get_config_read_mode())
            except TypeError as ex_path:
                try:
                    if isinstance(load_value(value), str):
                        raise ex_path
                    cfg_path = None
                    cfg_file = parse_string(parser, value, **kwargs)
                except (TypeError,) + get_loader_exceptions() as ex_str:
                    raise TypeError(f'Parser key "{dest}": {ex_str}') from ex_str
            else:
                cfg_file = parse_path(parser, value, **kwargs)
            cfg[dest].append(cfg_path)
            cfg.update(cfg_file)
