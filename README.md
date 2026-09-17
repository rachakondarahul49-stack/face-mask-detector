# 😷 Face Mask Detector

Real-time face mask classification with PyTorch, using a transfer-learned
MobileNetV2 backbone.

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project fine-tunes a MobileNetV2 (pretrained on ImageNet) to classify a
face image as **`with_mask`** or **`without_mask`**. It includes:

- A training pipeline with a proper train/validation split
- An evaluation script producing precision, recall, F1-score, and a confusion matrix
- A single-image prediction CLI
- A real-time webcam demo (OpenCV)

## Dataset

[Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
(Kaggle, by Omkar Gurav) — 7,553 images across two classes:

| Class          | Images |
|----------------|-------:|
| `with_mask`    |  3,725 |
| `without_mask` |  3,828 |

## Model

- **Backbone:** `torchvision.models.mobilenet_v2`, ImageNet-pretrained
- **Head:** final classifier layer replaced with `Linear(1280, 2)`
- **Input:** 224×224 RGB, ImageNet normalization
- **Loss / optimizer:** CrossEntropyLoss, Adam (`lr=1e-3`)
- **Training environment:** RTX 4050 Laptop GPU (CUDA)

## Results

Evaluated on a held-out validation split (20% of the dataset, 1,510 images)
during training, and on the full dataset with `src/evaluate.py`:

| Metric              | Held-out Val (best epoch) | Full-dataset eval |
|----------------------|:--------------------------:|:-------------------:|
| Accuracy             | 99.67%                    | 99.76%             |
| Precision (macro)    | —                          | 0.9976              |
| Recall (macro)       | —                          | 0.9976              |
| F1-score (macro)     | —                          | 0.9976              |

Confusion matrix (full-dataset eval):

|                  | Predicted: with_mask | Predicted: without_mask |
|------------------|:---------------------:|:-------------------------:|
| **Actual: with_mask**    | 3725                  | 0                         |
| **Actual: without_mask** | 18                    | 3810                      |

Only 18 misclassifications out of 7,553 images, all false negatives on the
without_mask class (predicted as wearing a mask when not).

## Project Structure

```
face-mask-detector/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   └── exploration.ipynb      # original prototyping notebook
├── src/
│   ├── model.py                # model definition (build/load)
│   ├── train.py                 # training loop with train/val split
│   ├── evaluate.py              # precision / recall / F1 / confusion matrix
│   ├── predict.py                # single-image inference CLI
│   └── webcam_demo.py            # real-time webcam demo
├── models/
│   └── face_mask_model.pth      # trained weights (not committed — see below)
└── samples/
    └── demo.gif / screenshots
```

## Setup

```bash
git clone https://github.com/rachakondarahul49-stack/face-mask-detector.git
cd face-mask-detector
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Get the dataset

```bash
kaggle datasets download -d omkargurav/face-mask-dataset
unzip face-mask-dataset.zip -d data
```

Expected layout:

```
data/
├── with_mask/
└── without_mask/
```

## Usage

**Train**

```bash
python src/train.py --data-dir data --epochs 10 --out models/face_mask_model.pth
```

**Evaluate on held-out data**

```bash
python src/evaluate.py --data-dir data --weights models/face_mask_model.pth
```

**Predict on a single image**

```bash
python src/predict.py --image samples/test.jpg --weights models/face_mask_model.pth
```

**Live webcam demo**

```bash
python src/webcam_demo.py --weights models/face_mask_model.pth
```

Trained weights aren't committed to this repo (see `.gitignore`) since
PyTorch checkpoints are large binary files. Either train your own with the
command above, or attach a release/Drive link here once you have one.

## Limitations & Future Work

- Classifies the **whole frame/image**, not individually detected faces —
  for multi-person frames, pair this with a face detector (e.g. OpenCV DNN
  or Haar cascade) and classify each cropped face.
- Trained only on frontal, mostly well-lit faces from the Kaggle dataset;
  accuracy on other angles, occlusions, or mask types is untested.
- No "incorrectly worn mask" class — binary only.
- Next steps: add a face-detection stage, expand to 3 classes, and quantify
  performance under low light / partial occlusion.

## Acknowledgments

- Dataset: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) by Omkar Gurav (Kaggle)
- Backbone: MobileNetV2 (Sandler et al., 2018), via `torchvision`

## License

MIT — see [LICENSE](LICENSE).
