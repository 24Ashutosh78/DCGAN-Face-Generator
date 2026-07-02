# from utils.dataset import get_dataloader

# import config

# loader = get_dataloader(

#     image_dir=config.DATASET_PATH,

#     batch_size=config.BATCH_SIZE
# )

# for images in loader:

#     print(images.shape)

#     break
#-----------------------------------------------------------
# import matplotlib.pyplot as plt

# from utils.dataset import get_dataloader

# import config

# loader = get_dataloader(

#     image_dir=config.DATASET_PATH,

#     batch_size=1
# )

# images = next(iter(loader))

# image = images[0]

# image = image.permute(1,2,0)

# image = (image + 1)/2

# plt.imshow(image)

# plt.axis("off")

# plt.show()

#--------------------------------------------------- generator.py 
# import torch

# from models.generator import Generator

# noise = torch.randn(8, 100, 1, 1)

# generator = Generator()

# fake_images = generator(noise)

# print("Noise Shape :", noise.shape)
# print("Output Shape:", fake_images.shape)

#----------------------------------------------------- Discriminatort.py
# import torch

# from models.discriminator import Discriminator

# # Create a fake batch of RGB images
# images = torch.randn(8, 3, 64, 64)

# # Initialize the Discriminator
# discriminator = Discriminator()

# # Forward pass
# output = discriminator(images)

# print("Input Shape :", images.shape)
# print("Output Shape:", output.shape)

#------------------Real Training COde
import torch
import torch.nn as nn
import torch.optim as optim

from models.generator import Generator
from models.discriminator import Discriminator

from utils.dataset import get_dataloader
from utils.weights import weights_init
from utils.visualization import save_generated_images
from utils.checkpoint import save_checkpoint

import config

# -----------------------------
# Data Loader
# -----------------------------
loader = get_dataloader(
    image_dir=config.DATASET_PATH,
    batch_size=config.BATCH_SIZE
)

# -----------------------------
# Models
# -----------------------------
generator = Generator(config.NOISE_DIM).to(config.DEVICE)
discriminator = Discriminator().to(config.DEVICE)

# -----------------------------
# Initialize Weights
# -----------------------------
generator.apply(weights_init)
discriminator.apply(weights_init)

# -----------------------------
# Loss Function
# -----------------------------
criterion = nn.BCELoss()

# -----------------------------
# Optimizers
# -----------------------------
optimizerG = optim.Adam(
    generator.parameters(),
    lr=config.LEARNING_RATE,
    betas=(config.BETA1, 0.999)
)

optimizerD = optim.Adam(
    discriminator.parameters(),
    lr=config.LEARNING_RATE,
    betas=(config.BETA1, 0.999)
)

# -----------------------------
# Fixed Noise
# -----------------------------
fixed_noise = torch.randn(
    64,
    config.NOISE_DIM,
    1,
    1,
    device=config.DEVICE
)

# =============================
# Training Loop
# =============================
G_losses = []
D_losses = []

print("Training Started...\n")

G_losses = []
D_losses = []

for epoch in range(config.NUM_EPOCHS):

    for batch_idx, real_images in enumerate(loader):

        real_images = real_images.to(config.DEVICE)
        batch_size = real_images.size(0)

        # -----------------------------
        # Train Discriminator
        # -----------------------------
        real_labels = torch.full((batch_size,), 0.9, device=config.DEVICE)
        fake_labels = torch.zeros(batch_size, device=config.DEVICE)

        optimizerD.zero_grad()

        real_output = discriminator(real_images).view(-1)
        loss_real = criterion(real_output, real_labels)

        noise = torch.randn(
            batch_size,
            config.NOISE_DIM,
            1,
            1,
            device=config.DEVICE
        )

        fake_images = generator(noise)

        fake_output = discriminator(fake_images.detach()).view(-1)
        loss_fake = criterion(fake_output, fake_labels)

        lossD = loss_real + loss_fake

        lossD.backward()
        optimizerD.step()

        # -----------------------------
        # Train Generator
        # -----------------------------
        optimizerG.zero_grad()

        output = discriminator(fake_images).view(-1)

        lossG = criterion(output, real_labels)

        lossG.backward()
        optimizerG.step()

        # Save losses
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        # Print progress
        print(
            f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
            f"Batch [{batch_idx+1}/{len(loader)}] "
            f"Loss D: {lossD.item():.4f} "
            f"Loss G: {lossG.item():.4f}"
        )

    # -------------------------------------
    # End of Epoch
    # -------------------------------------

    with torch.no_grad():
        fake = generator(fixed_noise)

    save_generated_images(fake, epoch)

    save_checkpoint(
        generator,
        discriminator,
        optimizerG,
        optimizerD,
        epoch
    )

# print("\nTraining Completed!")

print("\nTraining Completed!")