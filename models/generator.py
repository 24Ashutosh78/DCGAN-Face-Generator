import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100): #100 random numbers noise values
        super().__init__()

        
        self.model=nn.Sequential(
            #First Layer
            nn.ConvTranspose2d(
                noise_dim,
                1024,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            #Second Layer
            nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #3rd Layer
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            #4th Layer
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            #5th Layer
            nn.ConvTranspose2d(128,3,4,2,1,bias=False),
            nn.Tanh()
        )
    
    def forward(self,x):
        return self.model(x)