# Copyright (c) OpenMMLab. All rights reserved.

from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead as _LinearClsHead


@HEADS.register_module(force = True)
class LinearClsHead(_LinearClsHead):
    def simple_test(self, x, softmax = True, post_process = False, **kwargs):
        return super().simple_test(x, softmax = softmax, post_process = post_process)
