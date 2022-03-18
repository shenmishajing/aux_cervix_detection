# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import build_head
from torch import nn

from .image_with_label import ImageWithLabelClassifier


class ImageWithLabelNoFusionClassifier(ImageWithLabelClassifier):
    def __init__(self,
                 label_head = None,
                 use_attention = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if label_head is not None:
            self.label_head = build_head(label_head)

        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Linear(self.img_fc.out_features + self.label_fc.out_features, 1)

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
        label = self.label_fc(label.to(x.dtype))
        if self.use_attention:
            attention = torch.sigmoid(self.attention(torch.cat((x, label), dim = 1)))
            x = x * attention
            label = label * (1 - attention)
        losses = dict()
        losses['loss_cls'] = self.head.forward_train(x, gt_label, **kwargs)['loss']
        losses['loss_label'] = self.label_head.forward_train(label, gt_label, **kwargs)['loss']

        return losses

    def simple_test(self, img, label, img_metas = None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        x = self.img_fc(x)
        label = self.label_fc(label.to(x.dtype))
        if self.use_attention:
            attention = torch.sigmoid(self.attention(torch.cat((x, label), dim = 1)))
            x = x * attention
            label = label * (1 - attention)

        res_img = self.head.simple_test(x, **kwargs)
        res_label = self.label_head.simple_test(label, **kwargs)
        res = (res_img + res_label) / 2

        return res
