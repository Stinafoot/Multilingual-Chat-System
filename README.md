# Assignment 4 - Computer Vision Fundamentals (Convolution · Edge Detection · SIFT + SVM)

Three computer vision questions covering manual convolution operations, comparative edge detection, and image classification using classical feature extraction with SIFT and SVM.

![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![Python](https://img.shields.io/badge/Python-3.8+-blue)

---

## Q1 - Manual Convolution, Activation & Pooling (20 marks)

Manually applies convolution, ReLU, and max pooling on a 4×4 grayscale image.

**Input image:**
```
I = [[1, 2, 0, 1],
     [3, 1, 2, 2],
     [0, 1, 3, 1],
     [2, 2, 1, 0]]
```

**Kernel:**
```
K = [[ 1, 0],
     [-1, 1]]
```

**Steps:**
1. Valid convolution (stride 1) → 3×3 output
2. ReLU: `max(0, x)` applied element-wise
3. 2×2 max pooling (stride 1) → 2×2 output

All intermediate matrices are printed at each step.

---

## Q2 - Edge Detection Algorithm Comparison (20 marks)

Implements and visually compares six classical edge detectors on `Lena.jpeg`.

| Method | Approach | Characteristics |
|--------|----------|----------------|
| **Sobel** | First-order gradient (3×3) | Good noise tolerance, directional sensitivity |
| **Prewitt** | First-order gradient (3×3) | Similar to Sobel, uniform kernel weights |
| **Roberts Cross** | First-order diagonal (2×2) | Fast, fine detail, high noise sensitivity |
| **Laplacian of Gaussian (LoG)** | Second-order derivative | Detects strong and weak edges; can double-edge |
| **Difference of Gaussian (DoG)** | Approximation of LoG | Controllable via sigma parameters |
| **Canny** | Multi-stage pipeline | Best continuity, non-max suppression + hysteresis |

All six outputs are displayed in a single comparison figure. Analysis covers:
- Edge sharpness
- Noise sensitivity
- Weak edge detection
- Boundary continuity
- Visual clarity

---

## Q3 - SIFT + SVM Image Classification (20 marks)

Implements a Bag of Visual Words (BoVW) pipeline for scene classification.

**Dataset:** [Caltech-101 / Intel Image Classification](https://www.kaggle.com/datasets/imbikramsaha/caltech-101/data)
Categories: buildings, forest, glacier, mountain, sea, street

### Pipeline

```
Raw image
    │
Grayscale conversion
    │
SIFT keypoint detection
    │  Each image → n × 128-D descriptors
K-Means clustering (all training descriptors)
    │  Builds visual vocabulary {C₁, C₂, ..., Cₖ}
Histogram of visual words
    │  Each image → k-D frequency histogram H
SVM classifier (RBF kernel)
    │
Evaluation: Accuracy · Precision · Recall · F1 · Confusion Matrix
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| SIFT descriptor dim | 128 |
| Visual vocabulary size (k) | Tunable (e.g., 50–200) |
| Classifier | `sklearn.svm.SVC` |
| Max images per class | 10 (for CPU-constrained environments) |

---

## Requirements

```bash
pip install numpy opencv-python matplotlib scikit-learn
```

## Usage

```bash
# Place Lena.jpeg in the working directory
# Download Caltech-101 / Intel dataset from Kaggle
jupyter notebook Program5.ipynb
```

---

## Files

| File | Description |
|------|-------------|
| `Program5.ipynb` | Full notebook: Q1 manual ops · Q2 edge detection · Q3 SIFT+SVM |
| `Lena.jpeg` | Test image for edge detection (add to working directory) |
