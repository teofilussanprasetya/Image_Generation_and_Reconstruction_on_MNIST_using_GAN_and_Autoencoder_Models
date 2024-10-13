# Image Generation and Reconstruction on MNIST using GAN and Autoencoder Models

# Executive Summary:
This project implements deep learning techniques for **image generation** and **dimensionality reduction** using the **MNIST dataset**. The focus is on two primary models: an **autoencoder** for dimensionality reduction and a **Generative Adversarial Network (GAN)** for image generation. The models are evaluated using the **Structural Similarity Index (SSIM)** for the autoencoder and the **Fréchet Inception Distance (FID)** for the GAN. The autoencoder reduces the dimensionality of images from 784 to 128, while the GAN generates realistic handwritten digits from random noise.

# Dataset:
The project uses the [Fashion MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data), which consists of 60,000 training and 10,000 test images of fashion items.

# Business Problem:
The project addresses two critical challenges:
1. **Dimensionality Reduction**: Efficiently reducing image data size while preserving image quality, which is essential in areas like data compression and image analysis.
2. **Image Generation**: Generating realistic images from random noise to support creative fields, data augmentation, or creating synthetic datasets.

# Methodology:

### MNIST Dataset Preprocessing:
- **Data Scaling**: The dataset consists of 60,000 training and 10,000 test images, normalized to a pixel range of 0 to 1.
- **Data Splitting**: The data is split into 80% training, 10% validation, and 10% test sets for model training and evaluation.

### Autoencoder for Dimensionality Reduction:
- **Autoencoder Model**: A baseline autoencoder reduces the image dimensions from 784 to 128 latent dimensions.
- **Evaluation**: The **SSIM** metric evaluates image reconstruction quality by comparing the similarity between original and reconstructed images.
- **Model Tuning**: The architecture, including the number of layers and latent dimension size, is adjusted to improve the SSIM score.

### Generative Adversarial Network (GAN):
- **GAN Model**: A GAN is trained on the MNIST dataset, where the **generator** creates digit images from random noise, and the **discriminator** distinguishes between real and generated images.
- **Evaluation**: The **FID** score is used to measure the quality of generated images by comparing their distribution to real MNIST images.

# Skills:
- **Deep Learning Models**: Experience with building autoencoders and GANs using **TensorFlow** and **Keras**.
- **Python Programming**: Expertise with libraries like **NumPy**, **Matplotlib**, and **Keras** for model implementation and visualization.
- **Data Preprocessing**: Handling data normalization, scaling, and splitting, and managing input pipelines.
- **Model Evaluation**: Familiarity with advanced metrics like **SSIM** for image reconstruction and **FID** for image generation quality.
- **Hyperparameter Tuning**: Skilled in optimizing learning rates, latent dimensions, and other hyperparameters for model performance.

# Results:
- **Autoencoder Performance**: The autoencoder effectively reduced image dimensionality to 128 while maintaining a high **SSIM score**, indicating good reconstruction quality.
- **GAN Performance**: The GAN generated realistic images with a **FID score** of 2.45, although further improvements are possible.

# Business Recommendations:
1. **Image Compression**: The autoencoder can be applied in fields requiring high-quality image compression, such as healthcare or media.
2. **Synthetic Image Generation**: The GAN model could support creative industries or be used to augment training datasets in cases where real data is scarce or costly.

# Next Steps:
1. **Optimize GAN Performance**: Explore more advanced architectures, such as **DCGAN** or **StyleGAN**, and fine-tune hyperparameters to improve the quality of generated images.
2. **Refine Autoencoder**: Test deeper architectures or different bottleneck sizes to optimize the trade-off between compression and reconstruction quality.
3. **Transfer Learning**: Apply pre-trained models like **InceptionV3** to more complex image generation tasks beyond MNIST.
