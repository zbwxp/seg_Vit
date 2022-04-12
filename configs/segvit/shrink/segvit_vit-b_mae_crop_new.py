_base_ = [
    './segvit_vit-b_jx_crop_new.py'
]
checkpoint = './pretrained/vit_base_p16_mae.pth'
model = dict(
    pretrained=checkpoint,
)
