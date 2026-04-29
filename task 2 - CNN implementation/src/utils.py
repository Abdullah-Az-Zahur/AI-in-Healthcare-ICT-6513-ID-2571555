import os

def download_dataset():
    if not os.path.exists("data"):
        print("Downloading dataset...")
        os.system("kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images")
        os.system("unzip lung-and-colon-cancer-histopathological-images.zip -d data")
    else:
        print("Dataset already exists.")