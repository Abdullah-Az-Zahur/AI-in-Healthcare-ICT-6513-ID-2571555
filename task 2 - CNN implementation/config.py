import torch

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/dataset"
NUM_CLASSES = 2