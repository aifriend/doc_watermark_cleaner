# Document Watermark Cleaner

Conditional Generative Adversarial Network (cGAN) for document image enhancement, implementing the DE-GAN architecture for watermark removal, binarization, deblurring, and cleaning.

## Overview

A deep learning approach to document enhancement using paired training data (degraded → clean). The model learns to transform degraded document images into clean versions through adversarial training.

## Capabilities

| Task | Description |
|------|-------------|
| Watermark Removal | Remove overlaid watermarks from documents |
| Binarization | Convert grayscale/color documents to clean binary |
| Deblurring | Sharpen blurred document images |
| Cleaning | General noise and artifact removal |

## Project Structure

```
├── augmentation/        # Data augmentation utilities
├── common/              # Shared utilities and GAN architecture
│   ├── Generator.py     # Generator network
│   └── utils.py         # Helper functions
├── service/             # Service wrapper
├── predict.py           # Inference script
├── train_dn.py          # Training script (denoise)
├── train_wm.py          # Training script (watermark)
└── requirements.txt
```

## Tech Stack

- **Language:** Python 3
- **Framework:** TensorFlow 2.x
- **Architecture:** Conditional GAN (pix2pix-based)

## Requirements

- Python 3.6+
- TensorFlow 2.x
- GPU recommended for training

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Download Pretrained Weights
Download from the [Google Drive link](https://drive.google.com) and save to `./weights/`.

### Inference
```bash
# Watermark removal
python predict.py --task unwatermark --input ./data_wm --output ./results

# Binarization
python predict.py --task binarize --input ./input --output ./output

# Deblurring
python predict.py --task deblur --input ./input --output ./output
```

### Training
Prepare paired datasets (degraded + ground truth images), then:
```bash
python train_wm.py   # Train watermark removal
python train_dn.py   # Train denoising
```

## Citation

If you use this work, please cite the original paper:

> Souibgui, M. A., & Kessentini, Y. (2020). DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

## License

This project is for **academic and research purposes only**. For commercial use, please contact the repository maintainer.

See the [LICENSE](LICENSE) file for details.

## Author

**Jose** — [@aifriend](https://github.com/aifriend)
