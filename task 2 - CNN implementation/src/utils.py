import os
import subprocess
import zipfile


def download_dataset():
    if not os.path.exists("data"):
        print("Downloading dataset...")
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "andrewmvd/lung-and-colon-cancer-histopathological-images",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                "Dataset download failed. Ensure Kaggle API is configured with kaggle.json.\n"
                + result.stderr
            )

        os.makedirs("data", exist_ok=True)
        with zipfile.ZipFile(
            "lung-and-colon-cancer-histopathological-images.zip", "r"
        ) as zip_ref:
            zip_ref.extractall("data")

    else:
        print("Dataset already exists. Skipping download.")
