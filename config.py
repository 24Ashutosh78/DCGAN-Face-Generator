import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE=64
CHANNELS=3
BATCH_SIZE=128
NOISE_DIM=100
LEARNING_RATE=0.0002
NUM_EPOCHS=100

BETA1=0.5

DATASET_PATH="dataset/celeba/image_align_celeba"


