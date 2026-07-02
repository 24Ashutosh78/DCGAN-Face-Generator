import os

import torch

def save_checkpoint(generator,
                    discriminator,
                    optimizerG,
                    optimizerD,
                    epoch):

    os.makedirs("outputs/checkpoints", exist_ok=True)

    checkpoint = {

        "epoch": epoch,

        "generator": generator.state_dict(),

        "discriminator": discriminator.state_dict(),

        "optimizerG": optimizerG.state_dict(),

        "optimizerD": optimizerD.state_dict()

    }

    torch.save(
        checkpoint,
        "outputs/checkpoints/checkpoint.pth"
    )