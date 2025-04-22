# Seminar 14: Advanced Diffusion

This seminar folder contains materials for the fourteenth seminar in our **Deep Generative Models** course. In this session we dive into *advanced diffusion techniques* with **[Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)**, spanning pipelines, scheduler design, prompt engineering, and parameter‑efficient fine‑tuning.

## Seminar Overview

- **Topic**: Advanced Diffusion with Hugging Face Diffusers  
- **Objective**:  
  - **DDPM Fundamentals** – revisit the forward & reverse processes, noise‑prediction loss, and classifier‑free guidance.  
  - **Text‑to‑Image Pipelines** – run Stable Diffusion v2.1, optimise memory via attention‑slicing, and explore guidance‑scale trade‑offs.  
  - **Scheduler Exploration** – compare Euler‑Discrete, DPM‑Solver++, and DDIM for speed vs. fidelity.  
  - **Prompt Engineering & Compel** – weight sub‑prompts for fine‑grained concept control.  
  - **Latent‑Space Editing** – practise image‑to‑image, inpainting, and batch grids.  
  - **Parameter‑Efficient Fine‑Tuning** – load & fuse LoRA adapters for rapid style transfer.  
  - **Textual Inversion** – extend model vocabulary with new tokens from only a few images.  

## Key Concepts

1. **Diffusion‑Model Mathematics**  
   - *Forward diffusion*: adds Gaussian noise per schedule $\beta_t$.  
   - *Reverse process*: a U‑Net learns to predict noise $\boldsymbol\epsilon$; training minimises $\mathcal L_\text{simple}$.  
   - *Classifier‑Free Guidance*: interpolates conditional & unconditional predictions.
2. **Stable Diffusion Text‑to‑Image Pipeline**  
   CLIP encodes text → latent diffusion in $\mathbb R^{4\times H/8\times W/8}$ → V‑AE decodes to RGB.
3. **Sampling Schedulers**  
   - **Euler‑Discrete** – fast, robust (≈ 25 steps).  
   - **DPM‑Solver++** – higher‑order, converges in 10‑15 steps.  
   - **DDIM** – deterministic, perfectly repeatable.
4. **Batch Generation & Image Grids**  
   Vectorised prompts + a helper `image_grid()` for quick visual comparisons.
5. **Prompt Engineering with Compel**  
   Syntax `(token:weight)` scales each concept’s CLIP embedding without retraining.
6. **Image‑to‑Image & Inpainting**  
   - **Strength** controls fidelity vs. creativity.  
   - Binary masks confine edits to selected regions.
7. **LoRA Adaptation**  
   Adds a low‑rank term $BA/r$ to frozen weights—≈ 0.2 % new parameters, fusable at inference.
8. **Textual Inversion**  
   Learns a new token `<S*>` from a handful of images, expanding the model’s vocabulary.

## Useful Links
### Papers
- **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps (Lu et al., 2022)**: [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)
- **LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2022)**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion (Gal et al., 2022)**: [arXiv:2208.01618](https://arxiv.org/abs/2208.01618)

### Code & Docs
- **Diffusers Documentation**: https://huggingface.co/docs/diffusers/index
- **Compel Documentation**: https://github.com/damian0815/compel/tree/main/doc
