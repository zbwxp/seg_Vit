import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt
from .atm_head import trunc_normal_, constant_init, trunc_normal_init
from mmcv.cnn import build_norm_layer

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@HEADS.register_module()
class TPNHead(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            **kwargs,
    ):
        super(TPNHead, self).__init__(
            in_channels=in_channels, **kwargs)
        dim = embed_dims
        self.use_stages = use_stages
        self.image_size = img_size

        proj_norm = []
        input_proj = []
        tpn_layers = []
        for i in range(self.use_stages):
            # FC layer to change ch
            proj = nn.Linear(self.in_channels, dim)
            trunc_normal_(proj.weight, std=.02)
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            norm = nn.LayerNorm(dim)
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            tpn_layers.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = tpn_layers
        self.q = nn.Embedding((self.image_size // 16)**2, dim)

        delattr(self, 'conv_seg')
        self.conv_0 = nn.Conv2d(dim, 256, 1, 1)
        self.conv_1 = nn.Conv2d(256, self.num_classes, 1, 1)
        _, self.syncbn_fc_0 = build_norm_layer(dict(type='SyncBN', requires_grad=True), 256)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        x = []
        for stage_ in inputs[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        # do not reverse
        bs = x[0].size()[0]
        laterals = []
        maps_size = []

        q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)

        for idx, (x_, proj_, norm_, decoder_) in \
                enumerate(zip(x, self.input_proj, self.proj_norm, self.decoder)):
            lateral = norm_(proj_(x_))
            q = decoder_(q, lateral.transpose(0, 1))

        q = self.d3_to_d4(q.transpose(0, 1))
        out = self.gen_output(q)

        return out


    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 32 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def gen_output(self, t):
        out = self.conv_0(t)
        out = self.syncbn_fc_0(out)
        out = F.relu(out, inplace=True)
        out = self.conv_1(out)

        return out