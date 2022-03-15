# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from .image import ImageClassifier


class ImageWithLabelClassifier(ImageClassifier):
    def __init__(self,
                 img_in_channels = 2048,
                 img_out_channels = 1024,
                 label_in_channels = 2048,
                 label_out_channels = 1024,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_fc = nn.Linear(img_in_channels, img_out_channels)
        self.label_fc = nn.Linear(label_in_channels, label_out_channels)

    def forward_train(self, img, label, gt_label, **kwargs):
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
        if isinstance(x, tuple):
            x = x[-1]
        x = self.img_fc(x)
        label = self.label_fc(label)
        x = torch.cat((x, label), dim = 1)

        losses = dict()
        loss = self.head.forward_train(x, gt_label, **kwargs)

        losses.update(loss)

        return losses

    def simple_test(self, img, label, img_metas = None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        x = self.img_fc(x)
        label = self.label_fc(label)
        x = torch.cat((x, label), dim = 1)

        res = self.head.simple_test(x, **kwargs)

        return res
