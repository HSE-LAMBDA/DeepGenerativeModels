# Seminar 1: Autoencoders

This seminar folder contains the materials for the first seminar of our Deep Generative Models course, dedicated to **Autoencoders**. 

## Seminar Overview

- **Topic**: Introduction to Autoencoders (AEs)
- **Goal**: Understand how AEs learn a compressed representation of data by reconstructing the original input from a lower-dimensional latent space.
- **Variants Covered**:
  1. Simple Autoencoder
  2. Denoising Autoencoder (DAE)
  3. Sparse Autoencoder (SAE)

The accompanying Jupyter notebook (`autoencoders.ipynb`) explores these architectures using the MNIST dataset. You will learn how to:
1. Build and train a **Simple Autoencoder** with convolutional layers.
2. Implement a **Denoising Autoencoder** to improve robustness against noise.
3. Introduce **Sparsity** into the latent layer using KL divergence or L1 regularization.

## Key Concepts

1. **Encoder**: Maps input $x$ to a latent representation $z$.
2. **Decoder**: Maps the latent representation $z$ back to the original space to reconstruct $\hat{x}$.
3. **Loss Functions**:
   - **MSE (Mean Squared Error)**: Common for real-valued data.
   - **BCE/Logits**: Often used for binary or $[0,1]$-scaled data (like MNIST).
4. **Denoising**: Learn to reconstruct the *clean* image from *noisy* inputs, enhancing feature extraction.
5. **Sparse Regularization**: Encourages the autoencoder to learn minimal, distinct feature activations, often via:
   - **KL Divergence** (on intermediate activations)
   - **L1 penalty** (on latent representations)

## Useful Links

- **t-SNE** (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- **MNIST** dataset: http://yann.lecun.com/exdb/mnist/
- Bank, D., Koenigstein, N., Giryes, R. (2023). **Autoencoders**. In: Rokach, L., Maimon, O., Shmueli, E. (eds) Machine Learning for Data Science Handbook. Springer, Cham. https://arxiv.org/abs/1211.4246