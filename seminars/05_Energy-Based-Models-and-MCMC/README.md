# Seminar 05: Energy-Based Models & MCMC

This seminar folder contains the materials for the fifth seminar of our Deep Generative Models course, dedicated to **Energy-Based Models (EBMs)** and their applications.

This README accompanies the notebooks **EBM.ipynb** and **MCMC.ipynb**, which provide in-depth, hands-on explorations of training Deep Energy-Based Models and applying Markov Chain Monte Carlo (MCMC) methods using Pyro.

## Notebooks Overview

### **EBM.ipynb**
This notebook illustrates practical details such as:
- Implementing Contrastive Divergence with short-run MCMC (Langevin Dynamics),
- Stabilizing training via a sampling buffer,
- Leveraging score matching concepts,
- Performing out-of-distribution detection with EBMs.

### **MCMC.ipynb**
This notebook explores several sampling methods in Pyro, including:
- **Rejection Sampling**
- **Metropolis-Hastings Algorithm**
- **Langevin Sampling (MALA)**
- **Hamiltonian Monte Carlo (HMC)**
- **MCMC with the No-U-Turn Sampler (NUTS)**

## Seminar Overview

- **Topic**: Introduction to Energy-Based Models and MCMC Methods
- **Goal**: Understand how EBMs define a (generally unnormalized) distribution through an energy function and how to train such models without direct access to the partition function. We will also explore:
  1. **Score Matching** — learning the gradient of the log-density without explicit normalization.
  2. **Noise-Contrastive Estimation (NCE)** — reframing unsupervised density estimation as a binary classification task to bypass partition function computation.
  3. **Contrastive Divergence (CD)** — approximating gradients with short-run Markov Chain Monte Carlo (MCMC) to avoid the intractable partition function.
  4. **Advanced MCMC Methods** (e.g., **SGLD**, **AIS**) and sampling buffer tricks to stabilize and speed up training.
  5. **Pyro for MCMC** — Implementing probabilistic inference with advanced MCMC algorithms.

## Key Concepts

1. **Energy Function & Partition Function**  
   - An **energy** $E_\theta(\mathbf{x})$ is used to define unnormalized log-probabilities $\exp(-E_\theta(\mathbf{x}))$.  
   - The **partition function** $Z_\theta$ normalizes these probabilities but is often intractable to compute directly in high dimensions.

2. **Score Matching**  
   - Learns the gradient of the log-density directly.  
   - Involves *denoising* variants and requires no explicit partition function computation.

3. **Noise-Contrastive Estimation (NCE)**  
   - Treats density estimation as distinguishing real data from *noise* samples.  
   - Relies on a ratio $\frac{\exp(-E_\theta(\mathbf{x}))}{q_0(\mathbf{x})}$ to differentiate real vs. noise distributions.

4. **Contrastive Divergence (CD)**  
   - Maximizes data log-likelihood by approximating intractable expectations with MCMC samples from the current model.  
   - Often implemented with **short-run MCMC** (e.g., Langevin Dynamics).

5. **Sampling & MCMC**  
   - **Langevin Dynamics**: Iteratively refines samples with gradient steps + noise to approximate draws from the EBM.  
   - **Sampling Buffers**: Storing MCMC states between training steps can reduce the need for full re-initialization and stabilize learning.  
   - **Metropolis-Hastings Algorithm**: A Markov Chain Monte Carlo method for generating samples from complex distributions.
   - **Hamiltonian Monte Carlo (HMC)**: Uses Hamiltonian dynamics to propose new states, improving exploration.
   - **No-U-Turn Sampler (NUTS)**: An adaptive variant of HMC that auto-tunes step sizes for better efficiency.

6. **Applications**  
   - **Generative Modeling**: Learning a joint distribution over high-dimensional data (images, etc.).  
   - **Out-of-Distribution Detection**: EBMs can indicate whether a sample falls off the learned data manifold.  
   - **Energy-Based Classifiers**: A unified framework to handle both classification and anomaly detection.  
   - **Probabilistic Inference with Pyro**: Leveraging MCMC to perform Bayesian inference.

## Useful Links

- **A Tutorial on Energy-Based Learning (LeCun et al., 2006)**: [ResearchGate](https://www.researchgate.net/profile/Marcaurelio-Ranzato/publication/216792742_A_Tutorial_on_Energy-Based_Learning/links/0912f50c6862425435000000/A-Tutorial-on-Energy-Based-Learning.pdf)
- **Implicit Generation and Modeling with Energy Based Models (Du et al., 2019)**: [NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/378a063b8fdb1db941e34f4bde584c7d-Abstract.html)
- **Estimation of Non-Normalized Statistical Models by Score Matching (Hyvärinen, 2005)**: [JMLR](https://jmlr.org/papers/v6/hyvarinen05a.html)
- **A Connection Between Score Matching and Denoising Autoencoders (Vincent, 2011)**: [Neural Computation](https://ieeexplore.ieee.org/abstract/document/6795935)
- **Noise-contrastive estimation: A new estimation principle for unnormalized statistical models (Gutmann et al., 2010)**: [PMLR](https://proceedings.mlr.press/v9/gutmann10a.html)
- **Training Products of Experts by Minimizing Contrastive Divergence (Hinton, 2002)**: [Neural Computation](https://ieeexplore.ieee.org/abstract/document/6789337)
- **How to Train Your Energy-Based Models (Song et al., 2021)**: [arXiv:2101.03288](https://arxiv.org/abs/2101.03288)
- **Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One (Grathwohl et al., 2019)**: [arXiv:1912.03263](https://arxiv.org/abs/1912.03263)
