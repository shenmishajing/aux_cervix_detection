import os
import sys


def get_log_dir(trainer):
    if trainer.checkpoint_callback is not None and trainer.checkpoint_callback.dirpath is not None:
        log_dir = os.path.dirname(trainer.checkpoint_callback.dirpath)
    else:
        save_dir = trainer.default_root_dir
        version = trainer.logger.version if isinstance(trainer.logger.version, str) else f"version_{trainer.logger.version}"
        log_dir = os.path.join(save_dir, str(trainer.logger.name), version)
    return log_dir


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
