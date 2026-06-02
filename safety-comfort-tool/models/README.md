# Model weights

The Perceptual Observer needs the DeepLabV3+ (ResNet-101) checkpoint pretrained on
**Cityscapes**: `best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar` (~450 MB).

This file is **not committed to git** — GitHub rejects files larger than 100 MB, and
the checkpoint is a third-party artifact (DeepLabV3+, see `../DeepLabV3Plus/LICENSE`).
It is distributed as a release asset instead.

## Get it with one command
```bash
python download_weights.py
```
The file lands at
`models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar`
(where `config.WEIGHT_PATH` points).

## For the repository owner (one-time)
1. Create a GitHub **Release** on your repo and attach the `.pth.tar` file
   (Releases support assets up to 2 GB).
2. Put that asset URL into `download_weights.py` (the `WEIGHTS_URL` constant), or
   tell users to pass it via the `WEIGHTS_URL` environment variable.

Alternatively, download the original checkpoint from upstream
(https://github.com/VainF/DeepLabV3Plus-Pytorch, "Pretrained Models") and place it
in this folder manually.
