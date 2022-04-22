# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import BaseClassifier, build_head
from torch import nn

from .image_transformer import FusionTransformer


class LabelTransformerClassifier(BaseClassifier):
    def __init__(self,
                 head = None,
                 num_labels = 1024,
                 fusion_transformer_cfg = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if head is not None:
            self.head = build_head(head)
        self.fusion_transformer_cfg = fusion_transformer_cfg
        self.fusion_transformer = FusionTransformer(**fusion_transformer_cfg)
        self.embed_dims = self.fusion_transformer.transformers[0].embed_dims
        self.num_transformer = self.fusion_transformer.num_transformer
        assert self.num_transformer in [1, 2], 'ImageTransformerWithLabelClassifier only support one or two transformers'

        self.num_labels = num_labels
        self.label_fcs = nn.ModuleList([nn.Linear(1, self.embed_dims) for _ in range(num_labels)])

    @staticmethod
    def extract_cls_token(tokens):
        return [torch.cat([t[:, 0] for t in token], dim = -1) for token in tokens]

    def extract_label_token(self, label):
        label_token = []
        for i in range(self.num_labels):
            label_token.append(self.label_fcs[i](label[:, i, None]))
        label_token = torch.stack(label_token, dim = 1)

        if self.num_transformer == 2:
            label_token = [label_token[:, :self.num_labels // 2], label_token[:, self.num_labels // 2:]]
        elif self.num_transformer == 1:
            label_token = [label_token]
        else:
            raise NotImplementedError

        return label_token

    def token_forward(self, x = None, label = None):
        return self.fusion_transformer(self.fusion_transformer.embed_forward(self.extract_label_token(label)))

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

        tokens = self.token_forward(label = label.to(torch.float32))
        cls_token = self.extract_cls_token(tokens)[-1]

        losses = dict()
        loss = self.head.forward_train(cls_token, gt_label)

        losses.update(loss)

        return losses

    def simple_test(self, *args, label, **kwargs):
        """Test without augmentation."""

        tokens = self.token_forward(label = label.to(torch.float32))
        cls_token = self.extract_cls_token(tokens)[-1]

        res = self.head.simple_test(cls_token, **kwargs)

        return res

    def extract_feat(self, imgs, stage = None):
        pass
