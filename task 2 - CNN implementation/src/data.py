from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.utils import download_dataset
from src.config import IMG_SIZE, BATCH_SIZE

def get_loader():
    download_dataset()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder("data", transform=transform)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader