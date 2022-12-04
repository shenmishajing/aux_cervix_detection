import os
import sys


def get_log_dir(trainer):
    if trainer.checkpoint_callback is not None:
        if trainer.checkpoint_callback.dirpath is not None:
            dir_path = trainer.checkpoint_callback.dirpath
        else:
            dir_path = trainer.checkpoint_callback.__resolve_ckpt_dir(trainer)
        log_dir = os.path.dirname(dir_path)
    else:
        log_dir = trainer.default_root_dir
    return log_dir


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
