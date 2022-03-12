# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import build_head

from .image import ImageClassifier


class RotateImageClassifier(ImageClassifier):

    def __init__(self,
                 rotate_head = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if rotate_head is not None:
            self.rotate_head = build_head(rotate_head)

    def forward_train(self, img, gt_label, **kwargs):
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

        losses = dict(loss_cls = [], loss_rotate = [])
        features = []
        for i in range(4):
            x = self.extract_feat(kwargs['img_rotate'][:, i])
            features.append(x[-1])

            loss = self.head.forward_train(x, gt_label, **kwargs)
            losses['loss_cls'].append(loss['loss'])

            loss = self.rotate_head.forward_train(x, gt_label.new_full(gt_label.shape, i), **kwargs)
            losses['loss_rotate'].append(loss['loss'])

        losses['loss_cls'] = torch.stack(losses['loss_cls']).mean()
        losses['loss_rotate'] = torch.stack(losses['loss_rotate']).mean()
        features = torch.stack(features)
        features = features - torch.mean(features, dim = 0, keepdim = True)
        features = features.reshape(features.shape[0], -1)
        losses['loss_feature'] = torch.mean(torch.linalg.norm(features, dim = 1))
        losses['loss'] = torch.stack([v for v in losses.values()]).sum()

        return losses
