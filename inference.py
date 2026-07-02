import torch
import torchvision.utils as vutils
from models.generator import Generator
import config

#Model Load
generator = Generator(config.NOISE_DIM)
checkpoint = torch.load(
    "outputs/checkpoints/checkpoint.pth",
    map_location=config.DEVICE
)

generator.load_state_dict(checkpoint["generator"])

generator.to(config.DEVICE)

generator.eval()

#Face Generate
noise = torch.randn(64,config.NOISE_DIM,1,1,device=config.DEVICE)

with torch.no_grad():

    fake = generator(noise)

#Face Gener=erate
noise = torch.randn(64,config.NOISE_DIM,1,1,device=config.DEVICE)

with torch.no_grad():

    fake = generator(noise)

#Save
vutils.save_image(fake,"generated_faces.png",normalize=True,nrow=8)

