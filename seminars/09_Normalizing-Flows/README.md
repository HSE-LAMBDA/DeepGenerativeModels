# Seminar 09: Normalizing Flows

This seminar folder contains the materials for the ninth seminar of our Deep Generative Models course, focusing on normalizing flows. In this session, we explore how to transform a simple base distribution into a complex target distribution through a sequence of invertible transformations. This approach enables exact likelihood computation, flexible density estimation, and sample generation.

## Seminar Overview

- **Topic**: Normalizing Flows for Flexible Density Estimation  
- **Objective**:  
  - **Understanding Normalizing Flows**: Learn how invertible transformations, applied sequentially, map a simple base distribution (e.g., Gaussian or Logistic) to complex data distributions using the change-of-variables formula.
  - **Model Construction**: Build a normalizing flow model by integrating various flow components such as ActNorm, invertible 1x1 convolutions, and Neural Spline Flow Coupling layers.
  - **Training Process**: Train the model by maximizing the log-likelihood, which involves computing the prior probability and the sum of Jacobian determinants from the flow layers.
  - **Visualization Techniques**: Visualize both the forward (data → latent) and inverse (latent → data) mappings, examine intermediate flow transformations, and interpret the final density estimates.

## Key Concepts

1. **Normalizing Flows**  
   - **Definition**: A sequence of invertible transformations that convert a simple base distribution into a complex target distribution by using the change-of-variables formula.
   - **Mathematical Framework**: $\log p(z_K) = \log p(z_0) - \sum_{i=1}^{K} \log \left| \det \left( \frac{\partial f_i}{\partial z_{i-1}} \right) \right|$
   - **Significance**: This framework enables exact likelihood evaluation and flexible modeling of complex data distributions.

2. **Flow Transformations and Layers**  
   - **Neural Spline Flows (NSF)**: Utilize rational quadratic splines to achieve highly flexible, non-linear transformations.
   - **Invertible 1x1 Convolutions**: Permute and mix dimensions while ensuring invertibility, facilitating more expressive transformations.
   - **ActNorm Layers**: Normalize activations with learnable parameters to stabilize training.
   - **Coupling Layers**: Enable efficient computation of the Jacobian determinant by transforming part of the input conditioned on the rest.

3. **Training and Optimization**  
   - **Loss Function**: The negative log-likelihood (NLL) is minimized: $\text{loss} = -\sum \left( \log p(z_0) - \sum_{i=1}^{K} \log \left| \det \left( \frac{\partial f_i}{\partial z_{i-1}} \right) \right| \right)$
   - **Evaluation Strategy**: Analysis of both the forward (data → latent) and inverse (latent → data) mappings, along with visual inspection of the evolving density and grid warp.

4. **Visualization and Analysis**  
   - **Grid Warping**: Visualize how a regular grid is warped through successive flow layers to understand the contribution of each layer.
   - **Density Estimation**: Use scatter plots with a color map to represent the estimated density over the input space.

## Useful Links

- **Real NVP: Density Estimation using Real NVP (Dinh et al., 2016)**: [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- **Glow: Generative Flow with Invertible 1x1 Convolutions (Kingma et al., 2018)**: [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
- **Neural Spline Flows (Durkan et al., 2019)**: [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)
- **Normalizing Flows: An Introduction and Review of Current Methods (Kobyzev et al., 2019)**: [arXiv:1908.09257](https://arxiv.org/abs/1908.09257)
- **Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios at al., 2021)**: [JMLR](https://www.jmlr.org/papers/v22/19-1028.html)
- **pytorch-normalizing-flows GitHub Repository**: [https://github.com/karpathy/pytorch-normalizing-flows](https://github.com/karpathy/pytorch-normalizing-flows)
- **awesome-normalizing-flows GitHub Repository**: [https://github.com/janosh/awesome-normalizing-flows](https://github.com/janosh/awesome-normalizing-flows)
