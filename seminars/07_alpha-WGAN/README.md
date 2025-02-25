# Seminar 07: Alpha-WGAN

This seminar folder contains the materials for the seventh seminar of our Deep Generative Models course, focusing on Alpha-WGAN. In this session, we explore a hybrid generative model that combines the strengths of autoencoders and generative adversarial networks (GANs) under the robust framework of Wasserstein GAN with Gradient Penalty (WGAN-GP).

## Seminar Overview

- **Topic**: Alpha-WGAN – Bridging Autoencoders and GANs with WGAN-GP
- **Goal**:
    - **Autoencoding & Reconstruction**: Learn how an encoder maps images to a latent space and a generator reconstructs images from latent codes using $L_1$ reconstruction loss.
    - **Adversarial Training**: Understand the role of two discriminators—one for images and one for latent codes—in guiding the training process with adversarial losses.
    - **Latent Space Exploration**: Visualize the learned latent space through interpolation techniques (slerp and lerp) to assess the continuity and coherence of the latent representations.
- **Materials**:
    - `alpha-WGAN.ipynb`: Contains code for model definitions (Encoder, Generator, Image and Code Discriminators), training loop, evaluation utilities, and visualization functions.
    - **Data**: The model is trained on the CIFAR-10 dataset.

## Key Concepts
1. **Autoencoder & Latent Representations**
    - **Encoder**: Compresses input images into latent vectors.
    - **Generator**: Reconstructs images from the latent vectors.
    - **Reconstruction Loss**: Uses $L_1$ loss to measure the difference between original images and reconstructions.
2. **Generative Adversarial Networks (GANs)**
    - **Image Discriminator ($D$)**: Differentiates between real images and those generated or reconstructed by the model.
    - **Code Discriminator ($C$)**: Distinguishes between latent codes sampled from the prior and those encoded from real images.
    - **Adversarial Losses**: Both discriminators contribute to the overall training objective by enforcing realistic generation in both image and latent spaces.
3. **Wasserstein GAN with Gradient Penalty (WGAN-GP)**
    - **Wasserstein Loss**: Provides a smoother, more meaningful gradient for training compared to traditional GAN losses.
    - **Gradient Penalty**: Enforces Lipschitz continuity by penalizing deviations of the gradient norm from 1, which stabilizes training.
4. **Latent Space Interpolation**
    - **Spherical Linear Interpolation (slerp) & Linear Interpolation (lerp)**: Methods implemented to explore the smooth transitions in the latent space, demonstrating the model’s ability to generate semantically coherent interpolated images.
5. **Training Dynamics**
    - **Separate Optimizers**: Different optimizers for the autoencoder (encoder + generator) and the discriminators.
    - **Learning Rate Scheduling & Early Stopping**: Techniques employed to ensure stable convergence and prevent overfitting.
    - **Diagnostic Logging**: Extensive logging (using Comet.ml) of gradients and loss metrics to monitor training performance.

## Useful Links

- **Adversarial Autoencoders (Makhzani et al., 2015)**: [arXiv:1511.05644](https://arxiv.org/abs/1511.05644)
- **Variational Approaches for Auto-Encoding Generative Adversarial Networks (Rosca et al., 2017)**: [arXiv:1706.04987](https://arxiv.org/abs/1706.04987)
