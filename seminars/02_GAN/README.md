# Seminar 02: GANs

This seminar folder contains the materials for the second seminar of our Deep Generative Models course, dedicated to **Generative Adversarial Networks (GANs)** and several of their variants. Below you’ll find an overview, key concepts and useful references based on the provided code.

## Seminar Overview

- **Topic**: Introduction to Generative Adversarial Networks (GANs)
- **Goal**: Understand how adversarial training can learn to generate realistic-looking data by **pitting a Generator ($G$) and a Discriminator ($D$)** against each other in a min-max game.
- **Variants Covered**:
  1. **Unconditional GAN** (simple DCGAN-like approach)
  2. **Conditional GAN** (cGAN) with label embeddings
  3. **f-GAN** (generalized framework using different f-divergences)

The accompanying Jupyter notebook (`GANs.ipynb`) explores these architectures using:
1. **MNIST** (focusing on digit "0") for Unconditional and f-GAN examples.
2. **CIFAR100** (for cGAN) to demonstrate label-conditioned generation.

## Key Concepts

1. **Generative Adversarial Networks**:
   - Introduced by Ian Goodfellow et al. (2014).
   - Consist of a **Generator** $G$ and a **Discriminator** $D$.
   - **Min-Max Objective**:  
```math
 \min_G \max_D \; \mathcal{L}(G, D) = \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] \;+\; \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
```
   - The **Generator** learns to produce data that appear “real,” while the **Discriminator** learns to distinguish real data from generated (fake) data.

2. **Unconditional GAN**:
   - No additional information is provided to either network.
   - The **Generator** takes only random noise as input.
   - The **Discriminator** sees only an image (real or fake) and attempts to classify real vs. fake.

3. **Conditional GAN (cGAN)**:
   - Incorporates **labels** or auxiliary information into both $G$ and $D$.
   - The **Generator** receives noise + label embeddings and outputs class-specific images.
   - The **Discriminator** sees the image + label information to decide if they match and are real.

4. **f-GAN**:
   - A **generalized** perspective where different **f-divergences** (KLD, JS, Reverse KL, etc.) can be used.
   - The architecture is similar, but the **loss functions** differ by how we define $g(v)$ and its Fenchel conjugate $f^*(g(v))$.
   - Allows flexible experimentation beyond the classical GAN objective.

## Useful Links

- **GAN Original Paper (Goodfellow et al., 2014)**: [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)  
- **f-GAN (Nowozin et al., 2016)**: [arXiv:1606.00709](https://arxiv.org/abs/1606.00709)  
- **Unrolled GANs (Metz, et al., 2016)**: [arXiv:1611.02163](https://arxiv.org/abs/1611.02163)
- **DCGAN (Radford et al., 2015)**: [arXiv:1511.06434](https://arxiv.org/pdf/1511.06434)  
- **A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications (Gui et al., 2020)**: [arXiv:2001.06937](https://arxiv.org/abs/2001.06937)
- **A Survey on Generative Adversarial Networks: Variants, Applications, and Training (Jabbar et al., 2020)**: [arXiv:2006.05132](https://arxiv.org/abs/2006.05132)  
- **Improved Techniques for Training GANs (Salimans et al., 2016)**: [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)
- **How Generative Adversarial Networks and Their Variants Work: An Overview (Hong et al., 2017)**: [arXiv:1711.05914](https://arxiv.org/abs/1711.05914)  
- **Deconvolution and Checkerboard Artifacts**: [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)
- **MNIST** dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)  
- **CIFAR100** dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
