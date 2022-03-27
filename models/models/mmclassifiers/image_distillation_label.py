# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcls.models import build_backbone, build_loss
from torch import nn

from .image import ImageClassifier
from .image_with_label import ImageWithLabelClassifier


class ImageDistillationLabelTeacherClassifier(ImageWithLabelClassifier):
    def distillation_forward(self, img, label):
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            feat = x[-1]
        else:
            feat = x
        img_feat = self.img_fc(feat)
        label_feat = self.label_fc(label.to(feat.dtype))
        feat = torch.cat((img_feat, label_feat), dim = 1)
        return x, feat


class ImageDistillationLabelClassifier(ImageClassifier):
    def __init__(self,
                 label_backbone = None,
                 label_configs = None,
                 label_loss = dict(type = 'CrossEntropyLoss', loss_weight = 1.0),
                 distillation_classifier: nn.Module = None,
                 distillation_loss = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_classifier = distillation_classifier
        if self.distillation_classifier.init_cfg is not None:
            for p in self.distillation_classifier.parameters():
                p.requires_grad = False

        if distillation_loss is not None:
            self.distillation_loss = build_loss(distillation_loss)

        if label_backbone is not None:
            self.label_backbone = build_backbone(label_backbone)
        else:
            self.label_backbone = None

        self.distillation_img_fc = nn.Linear(self.distillation_classifier.img_fc.in_features,
                                             self.distillation_classifier.img_fc.out_features)
        self.distillation_label_fc = nn.Linear(self.distillation_classifier.img_fc.in_features,
                                               self.distillation_classifier.label_fc.out_features)

        self.learn_label = nn.ModuleList()
        self.label_loss = nn.ModuleList()
        self.label_configs = label_configs
        for cfg in label_configs:
            cfg['in_channels'] = self.distillation_classifier.label_fc.out_features
            self.learn_label.append(nn.Linear(self.distillation_classifier.label_fc.out_features, cfg['num_classes']))
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
            feat = x[-1]
        else:
            feat = x
        img_feat = self.distillation_img_fc(feat)

        if self.label_backbone is not None:
            label_feat = self.extract_label_feat(img)
            if isinstance(label_feat, tuple):
                label_feat = label_feat[-1]
            else:
                label_feat = label_feat
        else:
            label_feat = feat
        label_feat = self.distillation_label_fc(label_feat)

        feat = torch.cat([img_feat, label_feat], dim = 1)

        preds = self.head.simple_test(feat, **kwargs)

        distillation_x, distillation_feat = self.distillation_classifier.distillation_forward(img, label)
        distillation_preds = self.distillation_classifier.head.simple_test(distillation_feat, **kwargs)

        loss_label = []
        for i in range(len(self.label_configs)):
            loss_label.append(self.label_loss[i](self.learn_label[i](feat[:, -self.label_configs[i]['in_channels']:]), label[:, i]))

        losses = dict()
        losses['loss_label'] = torch.mean(torch.stack(loss_label))
        losses['loss_cls'] = self.head.loss(torch.log(preds), gt_label)['loss']
        losses['loss_distillation_cls'] = self.distillation_classifier.head.loss(torch.log(distillation_preds), gt_label)['loss']
        # losses['loss_distillation_backbone'] = torch.mean(
        #     torch.stack([self.distillation_loss(f, d_f) for f, d_f in zip(x[:-1], distillation_x[:-1])]))
        losses['loss_distillation_fc_feat'] = self.distillation_loss(feat, distillation_feat)
        losses['loss_distillation_preds'] = self.distillation_loss(preds, distillation_preds)

        return losses

    def simple_test(self, img, img_metas = None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            feat = x[-1]
        else:
            feat = x
        img_feat = self.distillation_img_fc(feat)

        if self.label_backbone is not None:
            label_feat = self.extract_label_feat(img)
            if isinstance(label_feat, tuple):
                label_feat = label_feat[-1]
            else:
                label_feat = label_feat
        else:
            label_feat = feat
        label_feat = self.distillation_label_fc(label_feat)

        feat = torch.cat([img_feat, label_feat], dim = 1)
        res = self.head.simple_test(feat, **kwargs)

        return res
