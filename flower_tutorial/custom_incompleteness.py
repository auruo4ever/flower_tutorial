import numpy as np
from PIL import Image
import random

# ---------- Incompleteness ----------
def make_incomplete_dataset(dataset, min_pct=0.0, max_pct=0.1, seed=42):
    """
    Randomly whitens a random percentage of images in a dataset.
    """
    random.seed(seed)
    percentage = random.uniform(min_pct, max_pct)
    n_total = len(dataset)
    n_incomplete = int(n_total * percentage)
    indices = set(random.sample(range(n_total), n_incomplete))

    def whiten_if_selected(example, idx):
        if idx in indices:
            img = example["img"]
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
            arr = np.array(img)
            arr[:] = 255  # whiten entire image
            example["img"] = Image.fromarray(arr.astype(np.uint8))
        return example

    dataset = dataset.map(whiten_if_selected, with_indices=True)
    return dataset, percentage


# ---------- Inaccuracy ----------
def make_inaccurate_dataset(dataset, min_pct=0.0, max_pct=0.1, num_classes=10, seed=42):
    """
    Randomly corrupts a percentage of labels in the dataset.
    """
    random.seed(seed)
    n_total = len(dataset)
    pct_inaccurate = random.uniform(min_pct, max_pct)
    n_inaccurate = int(n_total * pct_inaccurate)
    indices = set(random.sample(range(n_total), n_inaccurate))

    def corrupt_label(example, idx):
        if idx in indices:
            old_label = example["label"]
            new_label = random.choice([l for l in range(num_classes) if l != old_label])
            example["label"] = new_label
        return example

    dataset = dataset.map(corrupt_label, with_indices=True)
    return dataset, pct_inaccurate

