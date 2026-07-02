import os
from PIL import Image
import config

import torch 
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms

transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])

class CelebADataset(Dataset):
    def __init__(self,image_dir,transform=None):
        self.image_dir=image_dir
        self.transform=transform
        self.images=os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name=self.images[idx]
        img_path=os.path.join(self.image_dir,img_name)
        image=Image.open(img_path).convert("RGB")

        if self.transform:
            image=self.transform(image)

        return image
    


def get_dataloader(image_dir,batch_size):
    dataset=CelebADataset(
        image_dir=image_dir,
        transform=transform
    )
    print("Dataset path:", config.DATASET_PATH)
    print("Files in folder:", os.listdir(config.DATASET_PATH)[:10])
    loader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    return loader

        
