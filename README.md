#  DCGAN Face Generator using PyTorch

A Deep Convolutional Generative Adversarial Network (DCGAN) implemented **from scratch** using PyTorch to generate realistic human face images from random noise.

This project was built to understand the internal working of Generative Adversarial Networks (GANs), including adversarial training, Generator and Discriminator architectures, transposed convolutions, and training stability.

---

##  Project Overview

Generative Adversarial Networks (GANs) consist of two neural networks trained simultaneously:

- **Generator** – Learns to generate realistic human face images from random noise.
- **Discriminator** – Learns to distinguish between real and generated images.

Both networks compete against each other, enabling the Generator to progressively produce more realistic faces.

---

##  Features

-  Built completely from start using PyTorch
-  Custom CelebA dataset pipeline
-  Deep Convolutional GAN (DCGAN) architecture
-  Custom Generator implementation
-  Custom Discriminator implementation
-  DCGAN weight initialization
-  Adversarial training loop
-  Model checkpoint saving
-  Automatic image generation after every epoch
-  Loss visualization
-  Inference script for generating new faces

---

# 📂 Project Structure

```text
DCGAN-Face-Generator/

│
├── dataset/
│   └── celeba/
│
├── models/
│   ├── generator.py
│   └── discriminator.py
│
├── utils/
│   ├── dataset.py
│   ├── weights.py
│   ├── visualization.py
│   └── checkpoint.py
│
├── outputs/
│   ├── generated/
│   ├── checkpoints/
│   └── plots/
│
├── config.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

---

#  Architecture

```
                 Random Noise
                      │
                      ▼
               Generator Network
                      │
               Fake Face Images
                      │
                      ▼
            Discriminator Network
               │               │
           Real Face      Fake Face
```

---

#  Generator Architecture

```
Input Noise (100 × 1 × 1)

↓

ConvTranspose2D
1024 × 4 × 4

↓

ConvTranspose2D
512 × 8 × 8

↓

ConvTranspose2D
256 × 16 × 16

↓

ConvTranspose2D
128 × 32 × 32

↓

ConvTranspose2D

↓

Output Image
3 × 64 × 64
```

Activation Functions

- ReLU
- Batch Normalization
- Tanh (Output Layer)

---

#  Discriminator Architecture

```
Input Image
3 × 64 × 64

↓

Conv2D
64 × 32 × 32

↓

Conv2D
128 × 16 × 16

↓

Conv2D
256 × 8 × 8

↓

Conv2D
512 × 4 × 4

↓

Conv2D

↓

Real / Fake Probability
```

Activation Functions

- LeakyReLU
- Batch Normalization
- Sigmoid

---

#  Dataset

**CelebA Dataset**

- Celebrity Faces Dataset
- Original Size: **202,599 images**
- Image Size: **178 × 218**

For this implementation,

- Images were resized to **64 × 64**
- A subset of **2,000 images** was used for training due to CPU-only training constraints.

---

#  Hyperparameters

| Parameter | Value |
|-----------|------:|
| Image Size | 64 × 64 |
| Batch Size | 128 |
| Noise Dimension | 100 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Betas | (0.5, 0.999) |
| Epochs | 100 |
| Loss Function | Binary Cross Entropy (BCELoss) |

---

#  Training Results

The Generator progressively learned meaningful facial features during training.

Training observations:

- Early epochs generated random noise.
- Mid training produced blurry facial structures.
- Final epochs generated recognizable human faces.
- No significant mode collapse was observed.
- Stable Generator and Discriminator losses throughout training.

---

#  Results

### Generated Faces

> Add your `epoch_100.png` here.

```
outputs/generated/epoch_100.png
```

### Training Loss

> Add your generated loss plot here.

```
outputs/plots/loss.png
```

---

#  Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DCGAN-Face-Generator.git

cd DCGAN-Face-Generator
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Training

```bash
python train.py
```

---

#  Generate New Faces

```bash
python inference.py
```

Generated images will be saved as

```
generated_faces.png
```

---

#  Concepts Learned

During this project, I gained practical experience with:

- Generative Adversarial Networks (GANs)
- Deep Convolutional GAN (DCGAN)
- ConvTranspose2D
- Batch Normalization
- Weight Initialization
- Binary Cross Entropy Loss
- Adversarial Training
- PyTorch Dataset & DataLoader
- Model Checkpointing
- Inference Pipeline

---

#  Future Improvements

- Train on the complete CelebA dataset
- Label Smoothing
- BCEWithLogitsLoss
- Spectral Normalization
- TensorBoard Logging
- Mixed Precision Training
- Wasserstein GAN (WGAN)
- WGAN-GP
- Conditional GAN (cGAN)
- StyleGAN
git 

---

# 👨‍💻 Author

**Ashutosh Tiwari**

B.Tech Electronics & Computer Science

Interested in

- Deep Learning
- Computer Vision
- Generative AI
- Machine Learning
- Data Science

If you found this project helpful, consider giving it a ⭐ on GitHub.