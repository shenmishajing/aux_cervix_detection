# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models import BaseClassifier, build_head


class LabelClassifier(BaseClassifier):
    def __init__(self, head = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if head is not None:
            self.head = build_head(head)

    def forward_train(self, *args, label, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            label (Tensor): of shape (N, C) encoding input labels.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        loss = self.head.forward_train(label, gt_label)

        losses.update(loss)

        return losses

    def simple_test(self, *args, label, **kwargs):
        """Test without augmentation."""
        res = self.head.simple_test(label, **kwargs)

        return res

    def extract_feat(self, imgs, stage = None):
        pass
