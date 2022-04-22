# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmcls.models import build_head
from torch import nn

from .image_transformer import FusionTransformer
from .image_transformer_with_label import ImageTransformerWithLabelClassifier


class ImageTransformerMaeLearnLabelClassifier(ImageTransformerWithLabelClassifier):
    def __init__(self,
                 mask_label_token_num = 1,
                 decoder_transformer_cfg = None,
                 label_configs = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_transformer_cfg = decoder_transformer_cfg
        self.decoder_transformer = FusionTransformer(**decoder_transformer_cfg)
        self.decoder_embed_dims = self.decoder_transformer.transformers[0].embed_dims
        assert self.decoder_transformer.num_transformer == self.num_transformer, 'num_transformer must be equal'

        self.mask_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dims)) for _ in range(self.num_transformer)])
        self.mask_label_token_num = mask_label_token_num

        if isinstance(label_configs, dict):
            default_config = label_configs['default']
            label_configs = label_configs['configs']
            for cfg in label_configs:
                for k, v in default_config.items():
                    cfg.setdefault(k, v)
        self.learn_label = nn.ModuleList()
        self.label_configs = label_configs
        for cfg in label_configs:
            self.learn_label.append(build_head(cfg))

    @staticmethod
    def random_masking(xs, mask_token_num):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        xs: [N, L, D], sequence
        """
        N, L, D = xs[0].shape  # batch, length, dim
        len_keep = L - mask_token_num

        noise = torch.rand(N, L, device = xs[0].device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim = 1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim = 1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        xs_masked = [torch.gather(x, dim = 1, index = ids_keep.unsqueeze(-1).repeat(1, 1, D)) for x in xs]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device = xs[0].device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim = 1, index = ids_restore)

        return xs_masked, mask, ids_restore

    def token_forward(self, x, label):
        img_tokens = self.extract_img_token(x)
        label_tokens = self.extract_label_token(label)
        tokens = [torch.cat([label_token, img_token], dim = 1) for img_token, label_token in zip(img_tokens, label_tokens)]
        tokens = self.fusion_transformer.embed_forward(tokens)
        embed_label_tokens = [t[:, 1:1 + self.num_labels // 2] for t in tokens]
        embed_label_tokens, mask, ids_restore = self.random_masking(embed_label_tokens, self.mask_label_token_num)
        tokens = [torch.cat([token[:, :1], embed_label_token, token[:, 1 + ids_restore.shape[1]:]], dim = 1) for token, embed_label_token in
                  zip(tokens, embed_label_tokens)]
        token = self.fusion_transformer(tokens)[-1]

        mask_tokens = [self.mask_tokens[i].repeat(token[i].shape[0], ids_restore.shape[1] - embed_label_tokens[i].shape[1], 1) for i in
                       range(len(token))]
        tokens_ = [torch.cat([t[:, 1:1 + embed_label_tokens[0].shape[1]], mask_token], dim = 1) for t, mask_token in
                   zip(token, mask_tokens)]  # no cls token
        tokens_ = [torch.gather(token_, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, token[0].shape[2])) for token_ in
                   tokens_]  # unshuffle
        decoder_tokens = [torch.cat([t[:, :1, :], token_, t[:, 1 + embed_label_tokens[0].shape[1]:]], dim = 1) for t, token_ in
                          zip(token, tokens_)]  # append cls token
        decoder_tokens = self.decoder_transformer(decoder_tokens)[-1]
        label_token = torch.cat([t[:, 1:1 + ids_restore.shape[1]] for t in decoder_tokens], dim = 1)

        return token, label_token

    def test_token_forward(self, x):
        img_tokens = self.extract_img_token(x)
        tokens = [torch.cat([label_token, img_token], dim = 1) for img_token, label_token in zip(img_tokens, label_tokens)]
        tokens = self.fusion_transformer.embed_forward(tokens)
        embed_label_tokens = [t[:, 1:1 + self.num_labels // 2] for t in tokens]
        embed_label_tokens, mask, ids_restore = self.random_masking(embed_label_tokens, self.mask_label_token_num)
        tokens = [torch.cat([token[:, :1], embed_label_token, token[:, 1 + ids_restore.shape[1]:]], dim = 1) for token, embed_label_token in
                  zip(tokens, embed_label_tokens)]
        token = self.fusion_transformer(tokens)[-1]

        mask_tokens = [self.mask_tokens[i].repeat(token[i].shape[0], ids_restore.shape[1] - embed_label_tokens[i].shape[1], 1) for i in
                       range(len(token))]
        tokens_ = [torch.cat([t[:, 1:1 + embed_label_tokens[0].shape[1]], mask_token], dim = 1) for t, mask_token in
                   zip(token, mask_tokens)]  # no cls token
        tokens_ = [torch.gather(token_, dim = 1, index = ids_restore.unsqueeze(-1).repeat(1, 1, token[0].shape[2])) for token_ in
                   tokens_]  # unshuffle
        decoder_tokens = [torch.cat([t[:, :1, :], token_, t[:, 1 + embed_label_tokens[0].shape[1]:]], dim = 1) for t, token_ in
                          zip(token, tokens_)]  # append cls token
        decoder_tokens = self.decoder_transformer(decoder_tokens)[-1]
        label_token = torch.cat([t[:, 1:1 + ids_restore.shape[1]] for t in decoder_tokens], dim = 1)

        return token, label_token

        img_tokens = self.extract_img_token(x)
        mask_tokens = [self.mask_tokens[i].repeat(img_tokens[i].shape[0], self.num_labels // 2, 1) for i in range(len(img_tokens))]
        tokens = [torch.cat([label_token, img_token], dim = 1) for img_token, label_token in zip(img_tokens, mask_tokens)]
        tokens = self.fusion_transformer(self.fusion_transformer.embed_forward(tokens))
        return tokens

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

        token, label_token = self.token_forward(x, label.to(x.dtype))
        cls_token = self.extract_cls_token(token)

        loss_label = []
        for i in range(len(self.label_configs)):
            loss_label.append(self.learn_label[i].forward_train(label_token[:, i], label[:, i])['loss'])

        losses = dict()
        losses['loss_cls'] = self.head.forward_train(cls_token, gt_label, **kwargs)['loss']
        losses['loss_label'] = torch.mean(torch.stack(loss_label))
        return losses

    def simple_test(self, img, img_metas = None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if isinstance(x, tuple):
            x = x[-1]
        tokens = self.test_token_forward(x)
        cls_token = self.extract_cls_token(tokens[-1])

        return self.head.simple_test(cls_token, **kwargs)
