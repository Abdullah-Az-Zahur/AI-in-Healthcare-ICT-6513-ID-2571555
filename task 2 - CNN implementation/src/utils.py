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
        # Keep stdout/stderr attached so Kaggle CLI progress is visible.
        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                "Dataset download failed. Ensure Kaggle API is configured with kaggle.json.\n"
                "Also verify internet access and Kaggle credentials."
            )

        os.makedirs("data", exist_ok=True)
        with zipfile.ZipFile(
            "lung-and-colon-cancer-histopathological-images.zip", "r"
        ) as zip_ref:
            zip_ref.extractall("data")

    else:
        print("Dataset already exists. Skipping download.")
