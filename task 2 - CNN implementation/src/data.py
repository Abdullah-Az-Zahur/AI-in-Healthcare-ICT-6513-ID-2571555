import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import download_dataset
from config import BATCH_SIZE, IMG_SIZE


def _resolve_image_root(base_dir="data"):
    # Prefer lung-only subset so labels match NUM_CLASSES=3.
    preferred = os.path.join(base_dir, "lung_colon_image_set", "lung_image_sets")
    if os.path.isdir(preferred):
        return preferred

    if os.path.isdir(base_dir):
        return base_dir

    raise FileNotFoundError(f"Could not locate dataset directory under '{base_dir}'.")


def get_loader():
    download_dataset()

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )

    image_root = _resolve_image_root("data")
    dataset = datasets.ImageFolder(image_root, transform=transform)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader
