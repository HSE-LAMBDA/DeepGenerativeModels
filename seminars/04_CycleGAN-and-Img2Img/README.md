# Seminar 04: Image-to-Image

## Seminar Overview

In Seminar 04 of the Deep Generative Models course, we explore the task of image-to-image translation with a focus on unpaired translation techniques. This seminar will cover:
- The motivation and applications of unpaired image translation.
- Detailed insights into the CycleGAN architecture and its key innovations, such as cycle-consistency and identity mapping.
- A brief overview of StarGAN for multi-domain translation.
- A complete walkthrough of a CycleGAN implementation on the horse2zebra dataset using PyTorch.

## Key Concepts

- **Unpaired Image-to-Image Translation**: Learn to transform images from one domain to another without the need for paired datasets.
- **Cycle-Consistency Loss**: Ensures that an image translated from one domain to the other and back again remains close to the original image.
- **Identity Loss**: Helps preserve the color composition and key attributes when an image is already in the target style.
- **Adversarial Loss**: Drives the generator to produce outputs indistinguishable from real images in the target domain.

## Useful Links

### CycleGAN

- **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Zhu et al., 2017)**: [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
- **Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2016)**: [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
- **Github Reference**: [CycleGAN-PyTorch by Lornatang](https://github.com/Lornatang/CycleGAN-PyTorch)

### StarGAN

- **StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (Choi et al., 2017)**: [arXiv:1711.09020](https://arxiv.org/abs/1711.09020)
- **StarGAN v2: Diverse Image Synthesis for Multiple Domains (Choi et al., 2019)**: [arXiv:1912.01865](https://arxiv.org/abs/1912.01865)