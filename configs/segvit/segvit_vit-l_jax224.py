_base_ = [
    './segvit_vit-l_jax.py',
]
checkpoint = './pretrained/vit_large_p16_jax224.pth'
model = dict(
    pretrained=checkpoint,
)

data = dict(
    samples_per_gpu=4,)