# Seminar 10: Diffusion Probabilistic Models and Score Matching

This seminar folder contains the materials for the tenth seminar of our Deep Generative Models course. In this session, we dive into diffusion probabilistic models and various score matching techniques. The focus is on understanding how a forward diffusion process gradually corrupts data with noise and how a reverse process—parameterized by neural networks—learns to recover the original distribution. Additionally, we explore score matching methods and sampling via Langevin dynamics to estimate the gradients of the data density.

## Seminar Overview

- **Topic**: Diffusion Probabilistic Models and Score Matching  
- **Objective**:
  - **Forward Diffusion Process**: Learn how to gradually inject Gaussian noise into data to transform a complex data distribution into a simple, tractable prior.
  - **Reverse Process & Denoising**: Understand how to train a reverse Markov chain that recovers the original data from the noisy latent states. This includes the computation of posterior means and variances, and the variational inference framework based on the Evidence Lower Bound (ELBO).
  - **Score Matching**: 
    - **Standard Score Matching**: Estimate the score function (i.e., the gradient of the log-density) directly.
    - **Sliced and Denoising Score Matching**: Employ computationally efficient approximations and noise perturbation to improve scalability in high dimensions.
    - **Langevin Dynamics Sampling**: Generate samples by iteratively refining noisy inputs using the learned score function.
  - **Denoising Diffusion Probabilistic Models (DDPM)**: Explore the parameterization of the reverse process using noise prediction, fixed variance schedules, and training objectives that simplify the denoising task.

- **Practical Implementation**:  
  Two Jupyter notebooks illustrate these concepts:
  - `01_DPM_Score-Matching.ipynb`: Covers standard, sliced, and denoising score matching, including visualization of the learned score functions and sampling via Langevin dynamics.
  - `02_DPM_Models.ipynb`: Implements the forward diffusion process, reverse denoising process, and the DDPM framework with conditional models, complete with training loops and sampling procedures.

## Key Concepts

1. **Diffusion Probabilistic Models (DPM)**
   - **Forward Process**: A Markov chain that gradually adds Gaussian noise to data, transforming the original distribution into a nearly Gaussian prior.
   - **Reverse Process**: A learnable chain that inverts the forward process by estimating the posterior mean and variance, allowing data reconstruction.
   - **Variational Inference & ELBO**: Training is performed by optimizing an evidence lower bound that decomposes into KL divergence terms between the forward and reverse processes.

2. **Score Matching**
   - **Standard Score Matching**: Directly learns the score function, i.e., the gradients of the log-density.
   - **Sliced Score Matching**: Projects the gradient onto random directions to reduce computational cost.
   - **Denoising Score Matching (DSM)**: Incorporates noise injection into data to stabilize score estimation, especially in high-dimensional settings.
   - **Langevin Dynamics Sampling**: Utilizes the learned score function in a gradient-based iterative procedure to generate samples.

3. **Denoising Diffusion Probabilistic Models (DDPM)**
   - **Noise Prediction Parameterization**: Reformulates the reverse process to predict the noise component directly, simplifying the training objective.
   - **Variance Schedules**: Fixed schedules (e.g., linear, quadratic, sigmoid) for noise levels guide both the forward and reverse processes.
   - **Conditional Models & EMA**: Use conditional neural networks for denoising with timestep embeddings and apply exponential moving average (EMA) for training stability.

## Useful Links

- **Deep Unsupervised Learning using Nonequilibrium Thermodynamics (Sohl-Dickstein et al., 2015)**: [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
- **Denoising Diffusion Probabilistic Models (Ho et al., 2020)**: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- **Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2020)**: [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- **Improving and generalizing flow-based generative models with minibatch optimal transport (Tong et al., 2023)**: [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
- **A Connection Between Score Matching and Denoising Autoencoders (Vincent, 2011)**: [Neural Computation](https://direct.mit.edu/neco/article-abstract/23/7/1661/7677/A-Connection-Between-Score-Matching-and-Denoising)
- **Denoising Diffusion Implicit Models (Song, 2020)**: [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- **Estimation of Non-Normalized Statistical Models by Score Matching (Hyvärinen, 2005)**: [JMLR](https://jmlr.org/papers/v6/hyvarinen05a.html)
- **Sliced Score Matching: A Scalable Approach to Density and Score Estimation (Song et al., 2019)**: [arXiv:1905.07088](https://arxiv.org/abs/1905.07088)
- **Generative Modeling by Estimating Gradients of the Data Distribution (Song et al., 2019)**: [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
- **What are Diffusion Models? (Weng, 2021)**: [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)