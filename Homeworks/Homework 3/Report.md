# Homework 3 Report — Feature Spaces

## Overview

This homework explores correlation analysis and dimensionality reduction on the **Yale Face Database** — a collection of 2414 grayscale facial images (39 subjects × ~65 lighting conditions each). Each image is stored as a column of the matrix $\mathbf{X} \in \mathbb{R}^{1024 \times 2414}$, where the 1024 rows correspond to a 32×32 pixel image flattened into a vector.

---

## Exercise (a) — 100×100 Correlation Matrix

### Problem Statement

Compute a 100×100 correlation matrix $\mathbf{C}$ where each entry is the dot product between a pair of the first 100 images:

$$c_{jk} = \mathbf{x}_j^T \mathbf{x}_k$$

### Method

The double loop computes all pairwise dot products. Since each pixel value lies in $[0, 1]$, the dot product $\mathbf{x}_j^T \mathbf{x}_k$ is large when two images share similar bright regions and small when they differ (e.g., one is brightly lit and the other is dark).

### Results

The correlation matrix shows a clear block-diagonal structure. Within each block, images of the **same subject** under different lighting have high dot products (bright regions). Between blocks, the correlation drops, since different subjects have different facial geometry. This block structure is the signal that recognition algorithms exploit.

---

## Exercise (b) — Most and Least Correlated Image Pairs

### Problem Statement

Identify the most highly correlated and most uncorrelated pairs of distinct images from the 100×100 correlation matrix. Plot these faces.

### Method

To find pairs of *distinct* images, the diagonal of $\mathbf{C}$ is excluded (since $c_{jj} = \|\mathbf{x}_j\|^2$ is always the largest correlation for image $j$). We set diagonal entries to $-\infty$ before searching for the maximum and to $+\infty$ before searching for the minimum.

### Results

- **Most correlated pair:** images 86 and 88. These are photographs of the same subject under very similar lighting, so their pixel patterns are nearly identical.
- **Most uncorrelated pair:** images 54 and 64. One of these images (image 64) is nearly black (very low norm $\approx 0.02$), meaning it was captured under extreme lighting that leaves most pixels dark. Its dot product with any other image is close to zero.

This illustrates that the unnormalized dot product conflates **similarity** with **brightness**: a dark image will appear "uncorrelated" with everything, even itself. Normalizing by the norms (i.e., using cosine similarity) would separate these two effects.

---

## Exercise (c) — 10×10 Correlation Matrix for Selected Images

### Problem Statement

Compute the 10×10 correlation matrix for images at 1-indexed positions $[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]$.

### Method

The same dot-product computation as part (a), adjusting indices by $-1$ to convert from the problem's 1-based labelling to Python's 0-based indexing.

### Results

The 10×10 correlation matrix shows which of these ten hand-picked images are similar. Since the images span different subjects and lighting conditions across the full dataset (not just the first 100), the correlation values vary widely. Pairs from the same subject (e.g., images near indices 313 and 314) show high correlation, while images from different subjects or extreme lighting conditions show low correlation.

---

## Exercise (d) — Eigendecomposition of $\mathbf{Y} = \mathbf{X}\mathbf{X}^T$

### Problem Statement

Form the $1024 \times 1024$ matrix $\mathbf{Y} = \mathbf{X}\mathbf{X}^T$ and compute its six eigenvectors with the largest magnitude eigenvalues.

### Method

We use `numpy.linalg.eigh`, which exploits the fact that $\mathbf{Y}$ is real symmetric (and positive semi-definite). `eigh` returns eigenvalues in ascending order, so we sort by descending $|\lambda|$ and select the top six.

### Results

The six largest eigenvalues are:

| Rank | Eigenvalue |
|------|------------|
| 1 | 234,020 |
| 2 | 49,038 |
| 3 | 8,237 |
| 4 | 6,025 |
| 5 | 2,051 |
| 6 | 1,901 |

The steep drop from the first to the second eigenvalue (roughly 5:1) indicates that a single dominant mode — the average face illumination — captures the majority of the data's energy. The corresponding eigenvectors are the **eigenfaces**: basis images that span the principal subspace of the face dataset.

---

## Exercise (e) — SVD of $\mathbf{X}$

### Problem Statement

Compute the SVD of $\mathbf{X}$ and extract the first six principal component directions (left singular vectors).

### Method

`numpy.linalg.svd(X, full_matrices=False)` returns:

$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

The columns of $\mathbf{U}$ are the left singular vectors — the same principal directions that appear as eigenvectors of $\mathbf{X}\mathbf{X}^T$. The singular values $\sigma_i$ satisfy $\sigma_i^2 = \lambda_i$.

### Results

The first six singular values are $\sigma_1 = 483.8$, $\sigma_2 = 221.5$, $\sigma_3 = 90.8$, $\sigma_4 = 77.6$, $\sigma_5 = 45.3$, $\sigma_6 = 43.6$ (with $\sigma_1^2 \approx 234{,}020 = \lambda_1$, confirming consistency with part (d)).

---

## Exercise (f) — Eigenvector vs. SVD Mode Comparison

### Problem Statement

Compare $\mathbf{v}_1$ (first eigenvector from part (d)) with $\mathbf{u}_1$ (first left singular vector from part (e)) by computing $\| |\mathbf{v}_1| - |\mathbf{u}_1| \|_2$.

### Results

$$\| |\mathbf{v}_1| - |\mathbf{u}_1| \|_2 \approx 5.25 \times 10^{-16}$$

This is effectively **machine epsilon** — the two vectors are identical up to floating-point precision.

### Discussion

This result is expected from theory. If $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$, then

$$\mathbf{Y} = \mathbf{X}\mathbf{X}^T = \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^T$$

which is already an eigendecomposition of $\mathbf{Y}$ with eigenvalues $\sigma_i^2$ and eigenvectors $\mathbf{u}_i$. Therefore the eigenvectors of $\mathbf{Y}$ **are** the left singular vectors of $\mathbf{X}$, and the two approaches must agree (up to sign). We compare absolute values to eliminate the sign ambiguity, and the residual norm confirms exact agreement.

---

## Exercise (g) — Variance Captured by First 6 SVD Modes

### Problem Statement

Compute the percentage of total variance captured by each of the first 6 SVD modes and plot them as images.

### Method

The fraction of variance explained by mode $i$ is:

$$\text{Var}_i = \frac{\sigma_i^2}{\sum_{j} \sigma_j^2} \times 100\%$$

### Results

| Mode | Variance (%) |
|------|-------------|
| 1 | 72.93 |
| 2 | 15.28 |
| 3 | 2.57 |
| 4 | 1.88 |
| 5 | 0.64 |
| 6 | 0.59 |
| **Total** | **93.89** |

The first mode alone captures nearly **73%** of the total variance, and the first six modes together account for **93.9%**. This means a 2414-dimensional dataset can be represented to high fidelity in just a 6-dimensional subspace — a dramatic compression ratio of over 400:1.

The six SVD modes, when reshaped to 32×32 and visualized, are the classic **eigenfaces**:

- **Mode 1** represents the average illumination pattern — it is the dominant "bright face on dark background" template.
- **Modes 2–3** capture left-right and top-bottom lighting asymmetries.
- **Modes 4–6** encode finer facial features and expression-related variation.

---

## Summary

| Part | Key Result |
|------|-----------|
| (a) | 100×100 correlation matrix shows block-diagonal structure corresponding to subjects |
| (b) | Most correlated: images 86 & 88 (same subject, similar lighting); most uncorrelated: images 54 & 64 (one nearly black) |
| (c) | 10×10 correlation matrix for hand-picked images spans a wide range of similarity values |
| (d) | Top eigenvalue of $\mathbf{X}\mathbf{X}^T$ is 234,020, dominating all others |
| (e) | SVD singular values satisfy $\sigma_i^2 = \lambda_i$, confirming theoretical relationship |
| (f) | $\||\mathbf{v}_1| - |\mathbf{u}_1|\| \approx 10^{-16}$ — eigenvectors and SVD modes agree to machine precision |
| (g) | First 6 modes capture 93.9% of variance; the eigenfaces encode illumination and facial structure |

**Key takeaways:**

1. **Dot-product correlation** is a simple but effective similarity measure for images. Its block-diagonal structure on same-subject images is the foundation of nearest-neighbour face recognition.
2. **SVD and eigendecomposition are equivalent** routes to the same principal subspace. SVD of $\mathbf{X}$ is numerically preferable to forming $\mathbf{X}\mathbf{X}^T$ explicitly, since it avoids squaring the condition number.
3. **Low-rank structure is dramatic**: six modes out of 1024 possible dimensions capture over 93% of variance, validating the use of PCA / SVD as a dimensionality reduction tool for facial image data.
