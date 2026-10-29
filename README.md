# Fashion-MNIST Image Classification with a CNN

Deep Learning exercise: train a **Convolutional Neural Network (CNN)** in PyTorch to classify clothing images from the **Fashion-MNIST** dataset.

## Repository contents

| File | Description |
|---|---|
| `Fashion_MNIST_CNN_Exercise.ipynb` | Complete notebook: data preparation, CNN construction, training, evaluation, experiments and answers to the questions |
| `fashion_cnn.pth` | Trained weights of the baseline model (`FashionCNN`), saved with `torch.save(model.state_dict(), ...)` |
| `data/` | Created automatically on first run (Fashion-MNIST and CIFAR-10 downloads) |

## Dataset

Fashion-MNIST contains 70,000 grayscale images of **28 × 28 pixels**, split into 10 classes:

| Label | Class | Label | Class |
|---:|---|---:|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

Data split (seed `SEED = 42`):

- **Training**: 50,000 images
- **Validation**: 10,000 images (taken from the original training set)
- **Test**: 10,000 images, used only once at the end

## Installation

Python 3.9+ recommended.

```bash
pip install torch torchvision numpy matplotlib pandas jupyter
```

## Usage

```bash
jupyter notebook Fashion_MNIST_CNN_Exercise.ipynb
```

Run the cells **in order**, from top to bottom. The first run downloads the data into `./data` (the CIFAR-10 download at the end of the notebook can take a while).

### Reloading the saved model

```python
import torch
# The FashionCNN class must be defined (see section 10 of the notebook)
model = FashionCNN()
model.load_state_dict(torch.load("fashion_cnn.pth", map_location="cpu"))
model.eval()
```

## Baseline architecture

```text
Input             1 × 28 × 28
Conv2d(1→16, 3×3, padding=1) + ReLU + MaxPool(2)    → 16 × 14 × 14
Conv2d(16→32, 3×3, padding=1) + ReLU + MaxPool(2)   → 32 × 7 × 7
Flatten                                             → 1568
Linear(1568→64) + ReLU
Linear(64→10)                                       → 10 logits
```

Hyperparameters:

- **Loss function**: `CrossEntropyLoss` (multi-class classification, one correct class per image)
- **Optimizer**: `Adam`, learning rate `0.001`
- **Batch size**: 64
- **Epochs**: 10

A hidden layer of 128 neurons was tried first, but it caused overfitting, so it was reduced to 64.

## Results

### Baseline model

| Set | Accuracy | Loss |
|---|---:|---:|
| Training (epoch 10) | 93.15% | 0.1831 |
| Validation (epoch 10) | 91.46% | 0.2363 |
| **Test** | **90.78%** | 0.2557 |

### Per-class accuracy (test set)

| Class | Accuracy | Class | Accuracy |
|---|---:|---|---:|
| Trouser | 98.70% | Dress | 92.90% |
| Sandal | 98.20% | Pullover | 85.40% |
| Ankle boot | 98.20% | Coat | 83.70% |
| Bag | 97.80% | T-shirt/top | 83.00% |
| Sneaker | 93.70% | **Shirt** | **76.20%** |

The hardest class is **Shirt**: it is easily confused with T-shirt/top, Pullover and Coat, which have a very similar outline at this resolution.

### Experiments (one factor changed at a time)

| Experiment | Change | Train Acc. | Val. Acc. | Test Acc. |
|---|---|---:|---:|---:|
| Baseline | — | 93.15% | 91.46% | 90.78% |
| Experiment 1 | Dropout p = 0.5 | 91.08% | 90.99% | 90.42% |
| Experiment 2 | Conv2: 32 → 64 filters | 95.16% | 91.89% | 91.56% |
| Experiment 3 | Third conv layer (64 filters) | 94.89% | 91.86% | 91.01% |
| Experiment 4 | ReLU → GELU | 95.11% | 91.89% | 91.31% |

Experiments 1 to 4 use a hidden layer of 128 neurons.

**Conclusions:**

- Dropout almost closes the gap between training and validation accuracy, but does not improve validation accuracy: the baseline model was only slightly overfitting.
- Increasing the second convolution to 64 filters, or replacing ReLU with GELU, gives the best validation accuracy (91.89%), a modest gain (+0.4 points).

## Notebook outline

1. Imports and reproducibility
2. Downloading and visualizing Fashion-MNIST
3. Training / validation / test split and DataLoaders
4. Understanding the CNN: convolution, pooling, activations
5. Building and checking the model
6. Choosing the loss function and optimizer
7. Training and validation loops, loss and accuracy curves
8. Final test evaluation, per-class accuracy
9. Feature map visualization
10. Diagnosis: overfitting and underfitting
11. Regularization with dropout
12. Saving and reloading the model
13. Architecture experiments
14. Final questions
15. Next step: importing CIFAR-10 (3 × 32 × 32 color images)
