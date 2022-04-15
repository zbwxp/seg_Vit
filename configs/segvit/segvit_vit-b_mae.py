_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py'
]
checkpoint = './pretrained/vit_base_p16_mae.pth'
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        type="vit_fixpos",
    )
)
