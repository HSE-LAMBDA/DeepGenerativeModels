# Seminar 08: Autoregressive Models

This seminar folder contains the materials for the eighth seminar of our Deep Generative Models course, focusing on autoregressive image modeling with PixelCNN. In this session, we explore how to model the joint distribution of image pixels using masked convolutions, enabling efficient and coherent image generation.

## Seminar Overview

- **Topic**: PixelCNN for Autoregressive Image Modeling  
- **Objective**:  
  - **Understanding Autoregressive Models**: Learn how images can be modeled as a sequence of conditional distributions, where each pixel is predicted based on previously generated pixels.
  - **Data Preprocessing**: Discover techniques to bucketize continuous pixel values into discrete bins (0–9) and apply essential image transformations like resizing and grayscaling.
  - **Model Architecture**: Build a simple PixelCNN using masked convolutional layers (type A and B) to ensure that each pixel's prediction adheres to the autoregressive property.
  - **Autoregressive Sample Generation**: Implement a method to generate images pixel-by-pixel, starting from a partially completed image, using the trained PixelCNN model.
- **Content**:
  - Code for preprocessing images, including bucketizing pixel values.
  - Implementation of masked convolutions to maintain autoregressive constraints.
  - The PixelCNN model definition along with training and validation pipelines.
  - Autoregressive sampling demonstration to generate and visualize images.

## Key Concepts

1. **Autoregressive Modeling**  
   - **Definition**: Decomposing the joint probability of an image into a product of conditional probabilities:
     $$
     p(x) = \prod_{i=1}^{N} p(x_i \mid x_1, \ldots, x_{i-1})
     $$
   - **Importance**: This ensures that each pixel is generated based solely on the already generated pixels, a crucial aspect for coherent image synthesis.

2. **PixelCNN Architecture**  
   - **Masked Convolutions**:  
     - **Type A**: The center pixel is excluded from the convolution operation, ensuring that the prediction does not see the target pixel.
     - **Type B**: Allows the center pixel to be included, enabling deeper network architectures while maintaining the autoregressive property.
   - **Output**: Final logits represent a discrete probability distribution over 10 possible pixel buckets.

3. **Data Preprocessing & Transformations**  
   - **Bucketization**:  
     - Converts pixel values (0–255) into 10 discrete buckets using integer division and clamping.
   - **Image Transformations**:  
     - Resizing images to a lower resolution.
     - Converting images to grayscale.
     - Applying the bucketize function to map pixel intensities to discrete bins.

4. **Autoregressive Sample Generation**  
   - **Procedure**:  
     - Iteratively predict each pixel by conditioning on previously generated pixels.
     - Use softmax on the logits to sample from the categorical distribution for each pixel.
   - **Application**:  
     - Generate complete images from partially filled templates by sequentially sampling pixel values.

5. **Adapting to Color Images**  
   - **Separate Modeling**:  
     - Model each color channel (R, G, B) independently.
   - **Conditional Modeling**:  
     - Sequentially predict channels where, for example, the green channel is conditioned on the red channel and the blue on both red and green.  
   - **Architectural Changes**:  
     - Modify the network (e.g., using channel-specific masks) to capture inter-channel dependencies.

## Useful Links

- **Pixel Recurrent Neural Networks (van den Oord et al., 2016)**: [arXiv:1601.06759](https://arxiv.org/abs/1601.06759v3)  
- **PixelVAE: A Latent Variable Model for Natural Images (Gulrajani et al., 2016)**: [arXiv:1611.05013](https://arxiv.org/abs/1611.05013)  
- **Conditional Image Generation with PixelCNN Decoders (van den Oord et al., 2016)**: [NeurIPS](https://proceedings.neurips.cc/paper/2016/hash/b1301141feffabac455e1f90a7de2054-Abstract.html)  
- **PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications (Salimans et al. 2017)**: [arXiv:1701.05517](https://arxiv.org/abs/1701.05517)
