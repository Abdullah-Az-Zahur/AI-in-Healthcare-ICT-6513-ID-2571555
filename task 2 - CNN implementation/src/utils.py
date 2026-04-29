import os
import zipfile

def download_dataset():
    zip_file = "lung-and-colon-cancer-histopathological-images.zip"

    # Download
    if not os.path.exists(zip_file):
        print("Downloading dataset...")
        os.system("kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images")
    else:
        print("Zip already exists.")

    # Extract
    if not os.path.exists("data"):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("data")
    else:
        print("Dataset already extracted.")