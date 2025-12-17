import torch
from torch.utils.data import Sampler
import random
import numpy as np


class TaskBalancedBatchSampler(Sampler):
    """
    Fixed: Now strictly stops after one epoch worth of batches.
    """

    def __init__(self, dataset_task_ids, batch_size):
        self.batch_size = batch_size
        self.class_indices = {}

        # Group indices by task
        for idx, task_id in enumerate(dataset_task_ids):
            if task_id not in self.class_indices:
                self.class_indices[task_id] = []
            self.class_indices[task_id].append(idx)

        self.tasks = list(self.class_indices.keys())

        # Calculate Epoch Length (Total batches needed to cover largest class)
        # For your 50/50 split, this will be exactly 50 batches.
        max_len = max([len(v) for v in self.class_indices.values()])
        self.num_batches = (max_len // batch_size) * len(self.tasks)

    def __iter__(self):
        # 1. Shuffle indices INSIDE each class bucket at start of epoch
        for t in self.tasks:
            random.shuffle(self.class_indices[t])

        pointers = {t: 0 for t in self.tasks}
        total_yielded = 0  # <--- The Counter that fixes the infinite loop

        while total_yielded < self.num_batches:
            # Randomize task order for this round
            random.shuffle(self.tasks)

            for t in self.tasks:
                # Stop if we hit the limit
                if total_yielded >= self.num_batches:
                    return

                start = pointers[t]

                # Check if we run out of data in this bucket
                if start + self.batch_size > len(self.class_indices[t]):
                    # Reshuffle and loop back (Oversampling for small classes)
                    random.shuffle(self.class_indices[t])
                    pointers[t] = 0
                    start = 0

                # Yield the batch
                yield self.class_indices[t][start : start + self.batch_size]

                pointers[t] += self.batch_size
                total_yielded += 1

    def __len__(self):
        return self.num_batches
