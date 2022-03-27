# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.builder import build_head

from .image import ImageClassifier


class ImageDimlyClassifier(ImageClassifier):
    def __init__(self, dimly_head = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if dimly_head is not None:
            self.dimly_head = build_head(dimly_head)

    def forward_train(self, img, gt_label, gt_dimly_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        losses['loss_cls'] = self.head.forward_train(x, gt_label, **kwargs)['loss']
        losses['loss_label'] = self.dimly_head.forward_train(x, gt_dimly_label, **kwargs)['loss']

        return losses
