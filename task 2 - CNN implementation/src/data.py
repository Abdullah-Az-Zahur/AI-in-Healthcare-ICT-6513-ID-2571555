from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.utils import download_dataset
from src.config import IMG_SIZE, BATCH_SIZE

def get_loaders():
    download_dataset()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder("data", transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    return train_loader, val_loader