# Sources and licenses

`src/boq.py` and `src/backbones.py` are from Amar Ali-bey's
[Bag-of-Queries](https://github.com/amaralibey/Bag-of-Queries),
commit `1a4965ea7dfd9bd0dd846adf7a0e430f68101d12`. Their MIT license is
preserved in `LICENSE.boq`.

The initial independent AsymVPR implementation in `reproduction/` and its
tests were copied from the user's local `../Bag-of-Queries` project.
`results/training_history.json` records the original file hashes and training
provenance. This project is not the unavailable authors' AsymVPR repository.

The frozen gallery model uses the official BoQ DINOv2-B checkpoint:
https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth

DINOv2 source and pretrained model licenses apply separately; the local
configuration points to the existing DINOv2 checkout. EfficientViT-B2 is loaded
through timm's `efficientvit_b2.r224_in1k` implementation and ImageNet weights.
Dataset licenses and access conditions apply to the existing local datasets.
