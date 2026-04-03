# Homework 5: MNIST Classification with a Feedforward Neural Network

## The Big Picture

HW4 was about understanding **what the data looks like** (SVD) and then using **classical classifiers** (LDA, SVM, Decision Trees) to tell digits apart. HW5 asks you to do the same classification task, but replace those classifiers with a **Feedforward Neural Network (FNN)**.

The goal: build, train, and evaluate a simple FNN on MNIST, then compare its performance to what you got in HW4.

---

## Step 1: Load and preprocess the data (same as HW4)

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw = mnist.data.astype(float) / 255.0  # normalize to [0, 1]
y = mnist.target.astype(int)
```

**Why normalize?** Neural networks are sensitive to the scale of their inputs. Raw pixel values go from 0–255. Dividing by 255 puts them in [0,1], which makes gradient descent converge much faster and more stably.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=10000, random_state=42, stratify=y
)
```

`stratify=y` ensures every digit is equally represented in both splits.

---

## Step 2: (Optional but good) Apply PCA first

In HW4, you projected onto the top ~100 SVD modes before classifying. You can do the same here, or just feed the raw 784 pixels directly.

**For raw pixels:**
```python
X_train_input = X_train  # shape (60000, 784)
X_test_input  = X_test
```

**For PCA features (faster training, often similar accuracy):**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
X_train_input = pca.fit_transform(X_train)
X_test_input  = pca.transform(X_test)
```

**Why PCA first?** Fewer input dimensions = smaller network = faster training. But with a proper FNN you can also skip this entirely — the network learns its own features.

---

## Step 3: Understand what a Feedforward Neural Network is

Before writing any code, here's the mental model:

```
Input layer      Hidden layer 1     Hidden layer 2     Output layer
(784 neurons) → (256 neurons)   →  (128 neurons)   →  (10 neurons)
                 [ReLU]             [ReLU]              [Softmax]
```

- **Each layer** is a matrix multiplication: `output = activation(W @ input + b)`
- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`. It's the most common activation. It introduces non-linearity so the network can learn complex patterns.
- **Softmax** on the output: converts 10 raw scores into probabilities that sum to 1. The class with the highest probability is the prediction.
- **Training** means adjusting all the `W` and `b` parameters to minimize the loss (cross-entropy) using **backpropagation + gradient descent**.

---

## Step 4: Build the network with PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Convert numpy arrays to PyTorch tensors
X_tr = torch.FloatTensor(X_train_input)
y_tr = torch.LongTensor(y_train)
X_te = torch.FloatTensor(X_test_input)
y_te = torch.LongTensor(y_test)

# Define the network
class FNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)   # 10 output classes
        )
    
    def forward(self, x):
        return self.net(x)

model = FNN(input_dim=X_tr.shape[1])
```

**Why these sizes?** There's no single right answer. 256 and 128 are common choices that give a good balance between expressiveness and training speed. The output is always 10 (one per digit class).

---

## Step 5: Train the network

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()  # combines softmax + log-loss

# Mini-batch training
dataset = TensorDataset(X_tr, y_tr)
loader  = DataLoader(dataset, batch_size=256, shuffle=True)

train_losses = []
for epoch in range(20):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()          # clear old gradients
        preds = model(X_batch)         # forward pass
        loss  = loss_fn(preds, y_batch)
        loss.backward()                # backpropagation
        optimizer.step()               # update weights
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(loader))
    print(f"Epoch {epoch+1}/20 | Loss: {train_losses[-1]:.4f}")
```

Key vocabulary:
- **Epoch**: one full pass through the training data
- **Mini-batch**: instead of using all 60k samples at once, you use chunks of 256. This makes training faster and adds regularization.
- **Adam optimizer**: an improved version of gradient descent that adapts the learning rate automatically. Almost always a good default.
- **`optimizer.zero_grad()`**: PyTorch accumulates gradients by default; you reset them each batch.

---

## Step 6: Evaluate the model

```python
model.eval()  # turns off dropout/batchnorm if you have them
with torch.no_grad():  # no need to compute gradients for evaluation
    train_preds = model(X_tr).argmax(dim=1)
    test_preds  = model(X_te).argmax(dim=1)

train_acc = (train_preds == y_tr).float().mean().item()
test_acc  = (test_preds  == y_te).float().mean().item()
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test  accuracy: {test_acc:.4f}")
```

`.argmax(dim=1)` picks the highest-scoring class out of the 10 outputs.

---

## Step 7: Repeat for the hardest and easiest digit pairs (like HW4)

From HW4, you found:
- **Hardest pair**: 4 vs 9
- **Easiest pair**: 1 vs 6

Filter the data and train a binary FNN for each pair, just like you did for LDA/SVM:

```python
for d1, d2 in [(4, 9), (1, 6)]:
    mask_tr = np.isin(y_train, [d1, d2])
    mask_te = np.isin(y_test,  [d1, d2])
    # ... build tensors, train FNN with output_dim=2, evaluate
```

---

## Step 8: Compare to HW4 results

Make a table or bar chart:

| Classifier | All 10 (test) | 4 vs 9 (test) | 1 vs 6 (test) |
|---|---|---|---|
| LDA | 0.87 | 0.948 | 0.998 |
| SVM | 0.97 | 0.992 | 0.999 |
| FNN | ? | ? | ? |

This comparison is the core of the homework — you're showing what a neural network can do vs. classical methods.

---

## What to expect

A simple 2-layer FNN on MNIST should reach ~97–98% test accuracy on all 10 digits — better than LDA (~87%), similar to or slightly better than SVM (~97%). The main takeaway is: **the FNN learns its own features** through the hidden layers, which is why it's so powerful, even without PCA preprocessing.

---

## Summary of the steps

1. Load MNIST, normalize pixels to [0,1]
2. Split train/test (60k/10k)
3. (Optional) Apply PCA to reduce input to 100 dimensions
4. Define a simple FNN: Linear → ReLU → Linear → ReLU → Linear(10)
5. Train with Adam + CrossEntropyLoss for ~20 epochs using mini-batches
6. Evaluate: compute train and test accuracy
7. Repeat for the hardest (4 vs 9) and easiest (1 vs 6) pairs
8. Plot a loss curve and a comparison bar chart vs. HW4 classifiers

---

# Extra: Convolutional Neural Network (CNN)

## Why a CNN?

The FNN treats each pixel as an independent feature — it has no idea that pixels next to each other are spatially related. A **CNN** explicitly exploits spatial structure by sliding small filters (kernels) across the image to detect local patterns like edges, curves, and corners. This is why CNNs dominate image tasks.

The key insight: a digit "1" looks like a "1" whether it's slightly to the left or right. A CNN learns features that are **translation-invariant**; an FNN does not.

---

## Step E1: Reshape the data for the CNN

CNNs expect images in 2D form `(channels, height, width)`, not flattened vectors.

```python
# Reshape from (N, 784) to (N, 1, 28, 28)
# 1 = grayscale channel, 28x28 = image dimensions
X_tr_cnn = torch.FloatTensor(X_train).reshape(-1, 1, 28, 28)
X_te_cnn = torch.FloatTensor(X_test).reshape(-1, 1, 28, 28)
y_tr_cnn = torch.LongTensor(y_train)
y_te_cnn = torch.LongTensor(y_test)
```

**Important:** skip the PCA step here. CNNs need the 2D spatial structure of the image — PCA destroys that.

---

## Step E2: Understand the CNN building blocks

A CNN has two types of layers before the final classifier:

**Convolutional layer (`nn.Conv2d`):**
- Slides a small filter (e.g., 3×3) across the image
- Each filter detects one type of pattern (edge, curve, etc.)
- Multiple filters → multiple "feature maps"
- Parameters: `in_channels`, `out_channels` (number of filters), `kernel_size`

**Pooling layer (`nn.MaxPool2d`):**
- Shrinks the spatial size by taking the max value in each region
- Reduces computation and makes features more robust to small shifts
- A 2×2 max pool halves the height and width

```
Input (1, 28, 28)
  → Conv2d(1→32 filters, 3×3) + ReLU  → (32, 26, 26)
  → MaxPool2d(2×2)                     → (32, 13, 13)
  → Conv2d(32→64 filters, 3×3) + ReLU → (64, 11, 11)
  → MaxPool2d(2×2)                     → (64, 5, 5)
  → Flatten                            → (64*5*5 = 1600,)
  → Linear(1600, 128) + ReLU           → (128,)
  → Linear(128, 10)                    → (10,)
```

---

## Step E3: Build the CNN

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 1 input channel (grayscale), 32 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 28x28 -> 13x13
            nn.Conv2d(32, 64, kernel_size=3),  # 32 -> 64 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 13x13 -> 5x5
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                       # (64, 5, 5) -> 1600
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

cnn_model = CNN()
print(cnn_model)
```

---

## Step E4: Train the CNN (same loop as the FNN)

```python
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

dataset_cnn = TensorDataset(X_tr_cnn, y_tr_cnn)
loader_cnn  = DataLoader(dataset_cnn, batch_size=256, shuffle=True)

cnn_losses = []
for epoch in range(10):  # CNNs converge faster — 10 epochs is often enough
    cnn_model.train()
    epoch_loss = 0
    for X_batch, y_batch in loader_cnn:
        optimizer.zero_grad()
        preds = cnn_model(X_batch)
        loss  = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    cnn_losses.append(epoch_loss / len(loader_cnn))
    print(f"Epoch {epoch+1}/10 | Loss: {cnn_losses[-1]:.4f}")
```

The training loop is identical to the FNN — only the model and data shape differ. This is the beauty of PyTorch's abstraction.

---

## Step E5: Evaluate the CNN

```python
cnn_model.eval()
with torch.no_grad():
    cnn_train_preds = cnn_model(X_tr_cnn).argmax(dim=1)
    cnn_test_preds  = cnn_model(X_te_cnn).argmax(dim=1)

cnn_train_acc = (cnn_train_preds == y_tr_cnn).float().mean().item()
cnn_test_acc  = (cnn_test_preds  == y_te_cnn).float().mean().item()
print(f"CNN Train accuracy: {cnn_train_acc:.4f}")
print(f"CNN Test  accuracy: {cnn_test_acc:.4f}")
```

---

## Step E6: Hyperparameter tuning

The assignment asks you to **hyper parameter tune** both models. This means trying different configurations and picking the best. Things to try:

| Hyperparameter | What to vary | FNN example | CNN example |
|---|---|---|---|
| Learning rate | `lr` in Adam | 1e-2, 1e-3, 1e-4 | 1e-3, 5e-4 |
| Hidden layer size | neurons per layer | 64, 128, 256, 512 | 16, 32, 64 filters |
| Number of layers | depth | 1, 2, 3 hidden layers | 1, 2, 3 conv layers |
| Epochs | how long to train | 10, 20, 50 | 5, 10, 20 |
| Batch size | samples per update | 64, 128, 256 | same |
| Dropout | regularization | 0.2, 0.5 | same |

**Dropout** is worth adding — it randomly zeroes out neurons during training, which prevents overfitting:
```python
nn.Dropout(0.3)  # add this between layers
```

A simple way to tune: run a few combinations, plot train vs. test accuracy, and pick the model where test accuracy is highest without the train/test gap being too large (a big gap = overfitting).

---

## Step E7: Final comparison table

Fill this in after running everything:

| Model | All 10 (test) | 4 vs 9 (test) | 1 vs 6 (test) |
|---|---|---|---|
| LDA (HW4) | 0.871 | 0.948 | 0.998 |
| SVM (HW4) | 0.972 | 0.992 | 0.999 |
| Decision Tree (HW4) | 0.776 | 0.893 | 0.988 |
| FNN (HW5) | ? | ? | ? |
| CNN (HW5 Extra) | ? | ? | ? |

---

## What to expect from the CNN

A simple CNN like the one above typically reaches **~99% test accuracy** on MNIST — noticeably better than the FNN (~97–98%) and well above the classical methods. This is because:

1. **Fewer parameters** than an FNN on raw pixels (filters are shared across positions)
2. **Spatial awareness** — it knows that nearby pixels form meaningful patterns
3. **Hierarchical features** — early layers detect edges, later layers combine them into shapes

The CNN is the natural baseline for any image classification task.
