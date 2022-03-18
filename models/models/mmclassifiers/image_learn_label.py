# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcls.models import build_loss
from torch import nn

from .image import ImageClassifier


class ImageLearnLabelClassifier(ImageClassifier):
    def __init__(self,
                 pool_size = 3,
                 center_pool_size = 1,
                 in_channels = 2048,
                 img_out_channels = 1024,
                 label_out_channels = 1024,
                 label_configs = None,
                 label_loss = dict(type = 'CrossEntropyLoss', loss_weight = 1.0),
                 label_feat_use_expectation = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_size = pool_size
        self.center_pool_size = center_pool_size
        self.label_feat_use_expectation = label_feat_use_expectation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)

        self.learn_label = nn.ModuleList()
        self.label_loss = nn.ModuleList()
        self.label_configs = label_configs
        for cfg in label_configs:
            self.learn_label.append(nn.Linear(in_channels, cfg['num_classes']))
            loss_cfg = copy.deepcopy(label_loss)
            loss_cfg.update(cfg.get('loss', {}))
            self.label_loss.append(build_loss(loss_cfg))

        label_fc_in_channels = len(label_configs) if label_feat_use_expectation else sum([c['num_classes'] for c in label_configs])
        self.label_fc = nn.Linear(label_fc_in_channels, label_out_channels)
        self.img_fc = nn.Linear(in_channels, img_out_channels)

    def label_forward(self, x):
        label_feat = self.avg_pool(x)
        center_label_feat = torch.mean(
            label_feat[:, :, (self.pool_size - self.center_pool_size) // 2:(self.pool_size + self.center_pool_size) // 2], dim = [2, 3])
        round_label_feat = (torch.sum(label_feat, dim = [2, 3]) - center_label_feat * self.center_pool_size ** 2) / (
                self.pool_size ** 2 - self.center_pool_size ** 2)
        pred_label = []
        for i in range(len(self.label_configs)):
            if self.label_configs[i].get('center_feature', True):
                pred_label.append(self.learn_label[i](center_label_feat))
            else:
                pred_label.append(self.learn_label[i](round_label_feat))
        return pred_label

    def feature_forward(self, x):
        pred_label = self.label_forward(x)
        if self.label_feat_use_expectation:
            label_feat = []
            for p in pred_label:
                label_index = torch.arange(p.shape[1], device = p.device, dtype = p.dtype)
                label_feat.append(torch.sum(torch.softmax(p, dim = 1) * label_index[None, :], dim = 1))
            label_feat = torch.stack(label_feat, dim = -1)
        else:
            label_feat = torch.cat(pred_label, dim = 1)
        label_feat = self.label_fc(label_feat)
        img_feat = self.global_pool(x).squeeze()
        img_feat = self.img_fc(img_feat)
        x = torch.cat((img_feat, label_feat), dim = 1)
        return x, pred_label

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
        x, pred_label = self.feature_forward(x)
        loss_label = []
        for i in range(len(self.label_configs)):
            loss_label.append(self.label_loss[i](pred_label[i], label[:, i]))
        losses = dict()
        losses['loss_label'] = torch.mean(torch.stack(loss_label))
        losses['loss_cls'] = self.head.forward_train(x, gt_label, **kwargs)['loss']
        return losses

    def simple_test(self, img, img_metas = None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        x, _ = self.feature_forward(x)
        res = self.head.simple_test(x, **kwargs)

        return res
