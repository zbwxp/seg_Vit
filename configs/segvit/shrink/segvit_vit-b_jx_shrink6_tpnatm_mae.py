_base_ = [
    './segvit_vit-b_jx_shrink6_tpnatm.py'
]
checkpoint = './pretrained/vit_base_p16_mae.pth'
model = dict(
    pretrained=checkpoint,
)