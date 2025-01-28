# Seminar 03: Wasserstein GAN

This seminar folder contains the materials for the third seminar of our Deep Generative Models course, dedicated to **Wasserstein Generative Adversarial Networks (WGAN)** and its main variants.

## Seminar Overview

- **Topic**: Introduction to Wasserstein GANs
- **Goal**: Understand how using the **Wasserstein (Earth-Mover) distance** can stabilize GAN training and how to enforce the critic’s Lipschitz constraint using different techniques:
  1. **WGAN** (basic) — originally proposed with weight clipping.
  2. **WGAN-GP** — adds a **Gradient Penalty** to more smoothly enforce 1-Lipschitz.
  3. **WGAN + Spectral Normalization** — enforces 1-Lipschitz via spectral normalization.

## Key Concepts

1. **Wasserstein Distance / Earth-Mover Distance**  
   - Measures the “cost” of moving probability mass from the data distribution $p_{\text{data}}$ to the generator distribution $p_g$.  
   - Relies on the **Kantorovich-Rubinstein duality** for a practical loss function.

2. **WGAN Objective**  
   - Replaces the traditional discriminator with a **critic** that outputs a real value.  
   - Minimizes $\mathbb{E}[D(fake)] - \mathbb{E}[D(real)]$ under a **1-Lipschitz constraint** on $D$.

3. **Enforcing the 1-Lipschitz Constraint**  
   - **Weight Clipping** (original WGAN): Simple but can overly restrict the critic, leading to suboptimal performance.
   - **Gradient Penalty (WGAN-GP)**: Penalizes deviations from unit gradient norm on randomly interpolated data points ($\hat{x}$).
   - **Spectral Normalization**: Divides each layer’s weight by its largest singular value, ensuring Lipschitz continuity at each layer.

## Useful Links

### WGAN
- **Wasserstein GAN (Arjovsky, 2017)**: [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
- **Read-through: Wasserstein GAN (2017)**: [alexirpan.com/2017/02/22/wasserstein-gan.html](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)
- **Wasserstein GAN and the Kantorovich-Rubinstein Duality**: [vincentherrmann.github.io/blog/wasserstein/](https://vincentherrmann.github.io/blog/wasserstein/)
- **A Primer on Optimal Transport (Curuti, 2019)**:
   - [Part 1](https://www.youtube.com/watch?v=6iR1E6t1MMQ) 
   - [Part 2](https://www.youtube.com/watch?v=R49Xb9eAUBA) 
   - [Part 3](https://www.youtube.com/watch?v=SZHumKEhgtA) 
- **From GAN to WGAN (Weng, 2017)**: [lilianweng.github.io/posts/2017-08-20-gan/](https://lilianweng.github.io/posts/2017-08-20-gan/)

### WGAN-GP
- **Improved Training of Wasserstein GANs (Gulrajani et al., 2017)**: [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)  
- **GAN — Wasserstein GAN & WGAN-GP (Hui, 2018)**: [jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)

### General
- **Spectral Normalization for GAN (Miyato et al., 2018)**: [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)  
- **Instance Normalization: The Missing Ingredient for Fast Stylization (Ulyanov et al., 2016)**: [arXiv:1607.08022](https://arxiv.org/abs/1607.08022v3)
- **How Does Batch Normalization Help Optimization? (Santurkar et al., 2018)**: [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
- **CelebA (CelebFaces Attributes Dataset)**: [paperswithcode.com/dataset/celeba](https://paperswithcode.com/dataset/celeba)

   *CelebFaces Attributes dataset contains 202,599 face images of the size 178×218 from 10,177 celebrities, each annotated with 40 binary labels indicating facial attributes like hair color, gender and age.*
