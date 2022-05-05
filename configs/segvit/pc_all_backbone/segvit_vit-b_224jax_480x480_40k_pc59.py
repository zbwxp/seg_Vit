_base_ = [
    './segvit_vit-b_jax_480x480_40k_pc59.py',
]
checkpoint = './pretrained/vit_base_p16_jax_224.pth'
model = dict(
    pretrained=checkpoint,
)