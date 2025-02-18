# Seminar 06: Variational Autoencoders

This seminar folder contains the materials for the sixth seminar of our Deep Generative Models course, focusing on **Variational Autoencoders (VAEs)** and their advanced variants. We explore both the theoretical foundations behind VAEs—including variational inference, the reparameterization trick, and the Evidence Lower Bound (ELBO)—and practical implementations. In addition, we dive into a hierarchical variant, **VQ-VAE-2**, which leverages vector quantization and autoregressive priors for high-fidelity image generation.

This README accompanies two notebooks:
- **VAE.ipynb**: Demonstrates the implementation of a standard VAE on the CelebA dataset. It covers key concepts such as the reparameterization trick, KL divergence, and β-VAE for controlling disentanglement.
- **VQ-VAE-2.ipynb**: Provides a complete implementation of a two–stage VQ-VAE-2 model. It includes custom modules for vector quantization, residual blocks, and both top and bottom encoder/decoder architectures, as well as autoregressive PixelCNN priors for discrete latent code modeling.

## Seminar Overview

- **Topic**: Variational Autoencoders and their Variants
- **Goal**: 
  - Understand how VAEs overcome the intractability of direct posterior computation by learning an approximate posterior $q(\mathbf{z} \mid \mathbf{x})$ and maximizing the ELBO.
  - Explore the reparameterization trick that enables gradient-based optimization through stochastic nodes.
  - Investigate how β-VAE introduces a weighting factor to encourage disentangled latent representations.
  - Delve into VQ-VAE-2, a hierarchical architecture that discretizes latent spaces via vector quantization and employs autoregressive priors to generate realistic, diverse images.
- **Materials**:
  - **VAE.ipynb**: Covers the theory and practice behind variational inference, ELBO maximization, and image generation using VAEs on celebrity face data.
  - **VQ-VAE-2.ipynb**: Implements a two-stage VQ-VAE with both bottom and top latent representations, along with autoregressive models (PixelCNN) for sampling discrete latent codes.

## Key Concepts

1. **Variational Inference & ELBO**  
   - **Challenge**: Direct computation of $p(\mathbf{z} \mid \mathbf{x})$ is intractable due to high-dimensional integration.
   - **Solution**: Approximate the posterior with $q(\mathbf{z} \mid \mathbf{x})$ and maximize the Evidence Lower Bound (ELBO):
 ```math
 \log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z} \mid \mathbf{x})} \left[\log p(\mathbf{x} \mid \mathbf{z})\right] - D_{KL}\left(q(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z})\right)
 ```
   - This balances reconstruction accuracy with latent space regularization.

2. **Reparameterization Trick**  
   - Enables backpropagation through the stochastic sampling process by expressing $\mathbf{z}$ as:
 ```math
 \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
 ```
   - This reformulation is essential for efficient gradient-based optimization.

3. **KL Divergence & β-VAE**  
   - **KL Divergence**: Regularizes the latent distribution $q(\mathbf{z} \mid \mathbf{x})$ by minimizing its divergence from the prior $p(\mathbf{z})$, typically a standard normal distribution.
   - **β-VAE**: Introduces a hyperparameter $\beta$ to scale the KL divergence term:
```math
\mathcal{L}_{\beta\text{-VAE}} = \text{Reconstruction Loss} + \beta \cdot D_{KL}
```
     - $\beta = 1$ recovers the standard VAE.
     - $\beta > 1$ encourages a more disentangled latent space.

4. **VQ-VAE-2**  
   - **Vector Quantization**: Discretizes the latent space by mapping continuous representations to a finite set of embedding vectors.
   - **Hierarchical Architecture**: Uses a two-stage encoder/decoder (top and bottom) to capture coarse and fine-grained features.
   - **Autoregressive Priors**: Employs PixelCNN models to learn the distribution over discrete latent codes, enabling high-quality image generation.

## Useful Links

- **$\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework (Higgins et al., 2017)**: [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)
- **Understanding disentangling in $\beta$-VAE (Burgess et al., 2018)**: [arXiv:1804.03599](https://arxiv.org/abs/1804.03599)
- **Neural Discrete Representation Learning (van den Oord et al., 2017)**: [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- **Generating Diverse High-Fidelity Images with VQ-VAE-2 (Razavi et al., 2019)**: [arXiv:1906.00446](https://arxiv.org/abs/1906.00446)
- **Temporal Difference Variational Auto-Encoder (Gregor et al., 2018)**: [arXiv:1806.03107](https://arxiv.org/abs/1806.03107)
- **From Autoencoder to Beta-VAE (Weng, 2018)**: [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/)
- **Latent space cartography: Visual analysis of vector space embeddings (Yang et al., 2019)**: [Computer Graphics Forum](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13672)
- **Using Artificial Intelligence to Augment Human Intelligence (Carter et al., 2017)**: [Distill](https://distill.pub/2017/aia/)
