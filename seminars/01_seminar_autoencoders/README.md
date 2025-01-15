# Seminar 1: Autoencoders

This seminar folder contains the materials for the first seminar of our Deep Generative Models course, dedicated to **Autoencoders**.

## Seminar Overview

- **Topic**: Introduction to Autoencoders (AEs)
- **Goal**: Understand how AEs learn a compressed representation of data by reconstructing the original input from a lower-dimensional **latent space** and how to sample from this space for generating new data.
- **Variants Covered**:
  1. Simple Autoencoder
  2. Denoising Autoencoder (DAE)
  3. Sparse Autoencoder (SAE)

The accompanying Jupyter notebook (`autoencoders.ipynb`) explores these architectures using the MNIST dataset. You will learn how to:
1. Build and train a **Simple Autoencoder** with convolutional layers.
2. Implement a **Denoising Autoencoder** to improve robustness against noise.
3. Introduce **Sparsity** into the latent layer using KL divergence or L1 regularization.
4. **Sample from the latent space** to generate new data and visualize how the structure of the space reflects different data categories (e.g., digits).

## Key Concepts

1. **Latent Space**: A lower-dimensional hidden representation (often $\mathbf{z}$) where the essential features of the input $\mathbf{x}$ are captured. Learning and sampling from this space is at the core of generative tasks.
2. **Encoder**: A neural network module that maps the input $\mathbf{x}$ to a latent representation $\mathbf{z}$ in the latent space.
3. **Decoder**: A network module that reconstructs $\hat{\mathbf{x}}$ from the latent code $\mathbf{z}$, ideally recovering the original input.
4. **Sampling from Latent Space**: Using the learned distributions (mean and standard deviation) of activations in the latent space to generate new samples by sampling latent representations and decoding them into data.
5. **Loss Functions**:
   - **MSE (Mean Squared Error)**: A common choice for real-valued data reconstruction.
   - **BCE / Logits**: Often used for binary or $[0,1]$-scaled data (like MNIST).
6. **Denoising**: An approach in which the input is corrupted with noise, and the model learns to reconstruct the clean version, promoting more robust feature representations.
7. **Sparse Regularization**: Encourages the autoencoder to learn minimal, distinct feature activations, commonly through:
   - **KL Divergence** (on intermediate activations), or
   - **L1 penalty** (on latent representations).

## Useful Links

- **t-SNE** (scikit-learn): [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- **MNIST** dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Bank, D., Koenigstein, N., Giryes, R. (2023). **Autoencoders**. In: Rokach, L., Maimon, O., Shmueli, E. (eds) Machine Learning for Data Science Handbook. Springer, Cham. [https://arxiv.org/abs/1211.4246](https://arxiv.org/abs/1211.4246)
- [https://www.jeremyjordan.me/autoencoders/](https://www.jeremyjordan.me/autoencoders/)
