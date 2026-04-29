import os
import zipfile
import urllib.request
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_URL = "https://github.com/mohamedamine99/lung-cancer-dataset/archive/refs/heads/main.zip"

def download_dataset():
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/dataset"):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, "data/data.zip")

        with zipfile.ZipFile("data/data.zip", 'r') as zip_ref:
            zip_ref.extractall("data")

        os.rename("data/lung-cancer-dataset-main", "data/dataset")
    else:
        print("Dataset already exists")


class LungDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for label, folder in enumerate(os.listdir(root_dir)):
            path = os.path.join(root_dir, folder)

            if not os.path.isdir(path):
                continue

            for img in os.listdir(path):
                self.images.append(os.path.join(path, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])