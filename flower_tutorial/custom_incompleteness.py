import numpy as np
from PIL import Image
import random

def make_incomplete_dataset(dataset, min_pct=0.0, max_pct=0.4, seed=42):
    """
    Randomly whitens a random percentage of images in a Hugging Face dataset.
    
    Parameters:
    - dataset: HF Dataset (Flower partition)
    - min_pct, max_pct: bounds for random percentage (0-1)
    - seed: random seed for reproducibility
    """
    random.seed(seed)
    # Pick a random percentage within bounds
    percentage = random.uniform(min_pct, max_pct)
    n_total = len(dataset)
    n_incomplete = int(n_total * percentage)
    indices = set(random.sample(range(n_total), n_incomplete))

    def whiten_if_selected(example, idx):
        if idx in indices:
            img = example["img"]
            # Convert HF Arrow image to PIL
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
            arr = np.array(img)
            arr[:] = 255  # whiten entire image
            example["img"] = Image.fromarray(arr.astype(np.uint8))
        return example

    dataset = dataset.map(whiten_if_selected, with_indices=True)
    return dataset, percentage
