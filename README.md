# Fashion-MNIST Image Classification using Custom CNN

End-to-end Deep Learning Capstone Project: Designing, training, evaluating, and analyzing an honest failure case on Fashion-MNIST dataset using PyTorch.

---

## 📌 Project Overview
The goal of this project is to build a convolutional neural network (CNN) from scratch to classify 28x28 grayscale fashion images across 10 classes. The model is trained using clean train/validation/test splits, regularized to prevent overfitting, evaluated against a baseline model, and analyzed on failure cases.

### Dataset: Fashion-MNIST
* **Total size:** 70,000 grayscale images (28x28 pixels)
* **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
* **Honest Data Split:**
  * **Train Set:** 50,000 images (used for gradient updates)
  * **Validation Set:** 10,000 images (used for hyperparameter tuning & overfitting check)
  * **Test Set:** 10,000 images (held-out; touched only once for the final honest metric)

---

## 🏗 Model Architecture & Regularization

### 1. Custom CNN
The model follows the vision feature hierarchy (spatial grid shrinks while channels grow):
* **Conv Block 1:** Conv2D (1 -> 16, 3x3 kernel, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d (2x2, stride=2) `[28x28 -> 14x14]`
* **Conv Block 2:** Conv2D (16 -> 32, 3x3 kernel, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d (2x2, stride=2) `[14x14 -> 7x7]`
* **Classifier Head:** Flatten (32 * 7 * 7 = 1,568) -> Dropout (p=0.3) -> Linear (1568 -> 128) -> ReLU -> Dropout (p=0.3) -> Linear (128 -> 10)

### 2. Regularization Techniques
* **Data Augmentation:** `RandomCrop(28, padding=2)` and `RandomHorizontalFlip()` on the training set.
* **Dropout (0.3):** Applied before fully-connected layers to reduce co-adaptation.
* **Batch Normalization:** Stabilizes training dynamics and gradient flow.

---

## ⚙️ Hyperparameters & Reproducibility

To guarantee reproducibility, a fixed random seed is enforced:
* **Fixed Random Seed:** `42` (`torch.manual_seed(42)`, `np.random.seed(42)`)
* **Batch Size:** `64`
* **Optimizer:** `Adam` (lr=`0.001`)
* **Loss Function:** `nn.CrossEntropyLoss()` (computed directly on raw logits)
* **Epochs:** `10`
* **Hardware:** NVIDIA T4 GPU (Google Colab)

---

## 📊 Evaluation & Baseline Comparison

| Model | Technique / Regularization | Test Accuracy (%) | Test Loss |
| :--- | :--- | :---: | :---: |
| **Baseline (Logistic Regression / Small MLP)** | None | ~84.20% | ~0.450 |
| **Custom CNN (Ours)** | BatchNorm + Dropout + Data Aug | **90.11%** | **0.2888** |

*Note: Replace test metrics with your exact run numbers.*

### Learning Curves & Failure Analysis
* The training vs. validation loss curve shows stable convergence without severe overfitting gap.
* **Honest Failure Case:** The model misclassified a **Pullover** as a **Shirt** with 86.5% confidence. This confusion stems from high visual ambiguity in 28x28 grayscale resolution, where long sleeves, collar outlines, and fabric textures between knitwear/pullovers and shirts share almost identical pixel intensity patterns.

---

## 📁 Repository Structure

```text
├── README.md                 # Project documentation
├── requirements.txt          # Dependency list
├── project.ipynb             # Full end-to-end training & evaluation notebook
├── learning_curves.png       # Loss and accuracy curves
└── honest_failure_case.png   # Analysis of model failure case

```

## 🚀 How to Reproduce 

1. Clone the repository
   git clone https://github.com/RInabluuu/fashion-mnist-cnn-capstone.git

2. Install dependencies
   pip install -r requirements.txt

3. Run training and evaluation
   python main.py
   Or open and execute notebooks/capstone_experiment.ipynb on Google Colab with T4 GPU runtime enabled.
