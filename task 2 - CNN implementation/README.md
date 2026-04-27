# Lung Cancer CNN (GitHub + Colab Ready)

This project is structured for modular training and easy demonstration in class using Google Colab.

## Project Structure

- main.py: Entry point, calls pipeline functions only.
- src/config.py: Dataset and model config.
- src/data.py: Data loading.
- src/model.py: CNN model.
- src/train.py: Training loop.
- src/eval.py: Evaluation metrics.
- src/utils.py: Dataset download helper.
- notebooks/main.ipynb: Colab demo notebook.

## Run In VS Code

1. Install dependencies:
	pip install -r requirements.txt
2. Configure Kaggle API key (kaggle.json).
3. Run:
	python main.py

## Run In Colab (Recommended For Class Demo)

1. Open notebooks/main.ipynb from GitHub in Colab.
2. Set your GitHub repo URL in the first setup cell.
3. Upload kaggle.json when prompted.
4. Run all cells.

## Notes

- This project uses the Kaggle dataset:
  andrewmvd/lung-and-colon-cancer-histopathological-images
- Loader automatically prefers the lung-only subset so class count matches the model output.
