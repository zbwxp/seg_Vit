from mmseg.models.backbones.vit import VisionTransformer, TransformerEncoderLayer
from mmseg.models.builder import BACKBONES
import torch
import math
from mmcv.runner import ModuleList

@BACKBONES.register_module()
class vit_fixpos(VisionTransformer):
    def __init__(self, **kwargs):
        super(vit_fixpos, self).__init__(**kwargs)

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos = self.pos_embed.detach().clone()
        x = self._pos_embeding(x, hw_shape, pos)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                # no need to reshape
                # out = out.reshape(B, hw_shape[0], hw_shape[1],
                #                   C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)