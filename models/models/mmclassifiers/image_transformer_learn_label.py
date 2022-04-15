# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcls.models import build_backbone, build_loss
from torch import nn

from .image_transformer_with_label import ImageTransformerWithLabelClassifier


class ImageTransformerLearnLabelClassifier(ImageTransformerWithLabelClassifier):
    def __init__(self,
                 label_backbone,
                 learn_label_in_channels = 2048,
                 label_configs = None,
                 label_loss = dict(type = 'CrossEntropyLoss', loss_weight = 1.0),
                 label_feat_use_expectation = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_backbone = build_backbone(label_backbone)

        self.label_feat_use_expectation = label_feat_use_expectation
        self.learn_label = nn.ModuleList()
        self.label_loss = nn.ModuleList()
        self.label_configs = label_configs
        for cfg in label_configs:
            self.learn_label.append(nn.Linear(learn_label_in_channels, cfg['num_classes']))
            loss_cfg = copy.deepcopy(label_loss)
            loss_cfg.update(cfg.get('loss', {}))
            self.label_loss.append(build_loss(loss_cfg))

    def extract_label_feat(self, img, stage = 'neck'):
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.label_backbone(img)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x

    def label_forward(self, x):
        center_label_feat, round_label_feat = self.extract_center_round_feat(x)
        pred_label = []
        for i in range(len(self.label_configs)):
            if self.label_configs[i].get('center_feature', True):
                pred_label.append(self.learn_label[i](center_label_feat))
            else:
                pred_label.append(self.learn_label[i](round_label_feat))
        if self.label_feat_use_expectation:
            label_feat = []
            for p in pred_label:
                label_index = torch.arange(p.shape[1], device = p.device, dtype = p.dtype)
                label_feat.append(torch.sum(torch.softmax(p, dim = 1) * label_index[None, :], dim = 1))
            label_feat = torch.stack(label_feat, dim = -1)
        else:
            label_feat = torch.cat(pred_label, dim = 1)
        return pred_label, label_feat

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

        label_x = self.extract_label_feat(img)
        if isinstance(label_x, tuple):
            label_x = label_x[-1]

        pred_label, label_feat = self.label_forward(label_x)

        tokens = self.token_forward(x, label_feat)
        cls_token = self.extract_cls_token(tokens)[-1]

        loss_label = []
        for i in range(len(self.label_configs)):
            loss_label.append(self.label_loss[i](pred_label[i], label[:, i]))
        losses = dict()
        losses['loss_label'] = torch.mean(torch.stack(loss_label))
        losses['loss_cls'] = self.head.forward_train(cls_token, gt_label, **kwargs)['loss']

        return losses

    def simple_test(self, img, label, img_metas = None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        label_x = self.extract_label_feat(img)
        if isinstance(label_x, tuple):
            label_x = label_x[-1]

        _, label_feat = self.label_forward(label_x)

        tokens = self.token_forward(x, label_feat)
        cls_token = self.extract_cls_token(tokens)[-1]

        return self.head.simple_test(cls_token, **kwargs)
