
from datasets import load_dataset
import config

try:
    ds = load_dataset(config.DATASET_NAME, split=config.SPLIT)
    print("Dataset keys:", ds[0].keys())
    print("First item:", ds[0])
    if 'sentences' in ds[0]:
        print("Sentences example:", ds[0]['sentences'])
except Exception as e:
    print(f"Error loading dataset: {e}")
