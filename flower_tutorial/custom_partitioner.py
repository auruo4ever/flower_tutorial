import numpy as np
from typing import Dict, List
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner
import os

class NonIIDPartitioner(Partitioner):
    """Non-IID partitioner where each client gets some % of samples from one dominant class"""

    def __init__(
        self,
        num_partitions: int,
        samples_per_partition: int = 1000,
        dominant_ratios = [0.1, 0.6, 0.7, 0.2, 0.6, 0.7, 0.4, 0.7, 0.3, 0.6],
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self.samples_per_partition = samples_per_partition
        self.dominant_ratios = dominant_ratios
        self.seed = seed

        # Will store mapping: client_id -> list of indices
        self.partitions: Dict[int, np.ndarray] = {}

    @property
    def num_partitions(self) -> int:
        """Total number of partitions (clients)."""
        return self._num_partitions

    def _create_partitions(self) -> None:
        """Create the Non-IID partitions once the dataset is assigned."""
        if not self.is_dataset_assigned():
            raise RuntimeError("Dataset must be assigned before creating partitions.")
    
        rng = np.random.default_rng(self.seed)
        labels = np.array(self.dataset["label"])
        num_classes = len(set(labels))
        class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

        # Shuffle each class’s samples
        for c in class_indices:
            rng.shuffle(class_indices[c])

        classes = np.arange(num_classes)
        dominant_classes = rng.permutation(classes)  

        for client_id in range(self.num_partitions):
            # !!!! num_partitions < num_classes
            dominant_ratio = self.dominant_ratios[client_id]  
            dominant_class = dominant_classes[client_id % len(classes)]
            num_dominant = int(self.samples_per_partition * dominant_ratio)
            num_others = self.samples_per_partition - num_dominant

            # Get samples from dominant class
            dominant_samples = class_indices[dominant_class][:num_dominant]
            class_indices[dominant_class] = class_indices[dominant_class][num_dominant:]

            # Randomly get samples from other classes
            other_classes = classes[classes != dominant_class]
            other_indices: List[int] = []
            while len(other_indices) < num_others:
                c = rng.choice(other_classes)
                if len(class_indices[c]) == 0:
                    continue
                other_indices.append(class_indices[c][0])
                class_indices[c] = class_indices[c][1:]

            client_indices = np.concatenate([dominant_samples, other_indices])
            
            ######################
            # Just for debugging: print class distribution for this client
            unique, counts = np.unique(labels[client_indices], return_counts=True)
            class_distribution = dict(zip(unique, counts))

            log_path = "all_clients_distribution.txt"
            with open(log_path, "w") as f: 
                f.write(f"[PID {os.getpid()}] Client {client_id} | Dominant class: {dominant_class}\n")
                for cls, cnt in class_distribution.items():
                    f.write(f"  Class {cls}: {cnt:4d} samples\n")
                f.write(f"  Total: {len(client_indices)} samples\n")
                f.write("-" * 40 + "\n")
            ######################
            
            rng.shuffle(client_indices)
            self.partitions[client_id] = client_indices

    def load_partition(self, partition_id: int) -> Dataset:
        """Load a dataset partition for a given client ID."""
        if not self.partitions:
            self._create_partitions()

        if partition_id not in self.partitions:
            raise ValueError(f"Invalid partition_id {partition_id}")

        indices = self.partitions[partition_id]
        return self.dataset.select(indices)
