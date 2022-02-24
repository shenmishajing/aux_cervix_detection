from mmcv import ConfigDict

from mmdet.models.detectors import TwoStageDetector as _TwoStageDetector


class TwoStageDetector(_TwoStageDetector):
    def __init__(self, *args, **kwargs):
        args = [ConfigDict(a) for a in args]
        kwargs = {k: ConfigDict(v) for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)
