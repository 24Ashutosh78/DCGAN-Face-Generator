import os

import torchvision.utils as vutils

def save_generated_images(images, epoch):

    os.makedirs("outputs/generated", exist_ok=True)

    filename = f"outputs/generated/epoch_{epoch+1}.png"

    vutils.save_image(
        images,
        filename,
        normalize=True,
        nrow=8
    )