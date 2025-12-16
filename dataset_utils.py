from torch.utils.data import Sampler
import random


class TaskBalancedBatchSampler(Sampler):
    """
    A Custom Sampler for Multi-Task Learning.

    Problem: The PrPSeg architecture requires all images in a single batch
             to belong to the SAME Task (Class) to generate the correct tokens.

    Solution: This sampler organizes indices by class. It yields a batch of
              Class A, then a batch of Class B, etc., in a round-robin fashion.
    """

    def __init__(self, dataset_task_ids, batch_size):
        """
        Args:
            dataset_task_ids: A list of class IDs corresponding to the dataset.
                              e.g. [0, 0, 1, 2, 0, ...]
            batch_size: The size of the batch (e.g. 4)
        """
        self.batch_size = batch_size
        self.class_indices = {}

        # 1. Group all image indices by their Task ID
        # Result: {0: [idx1, idx5...], 1: [idx2, idx3...]}
        for idx, task_id in enumerate(dataset_task_ids):
            if task_id not in self.class_indices:
                self.class_indices[task_id] = []
            self.class_indices[task_id].append(idx)

        self.tasks = list(self.class_indices.keys())

        # 2. Calculate Epoch Length
        # We define one epoch as roughly covering all images.
        # We calculate how many batches the largest class would generate,
        # and multiply by number of classes to ensure we loop enough times.
        max_len = max([len(v) for v in self.class_indices.values()])
        self.num_batches = (max_len // batch_size) * len(self.tasks)

    def __iter__(self):
        # 1. Shuffle indices INSIDE each class bucket
        # This ensures we don't see the same "Proximal Tubule" images in the same order.
        for t in self.tasks:
            random.shuffle(self.class_indices[t])

        # Pointers track our progress through each bucket
        pointers = {t: 0 for t in self.tasks}

        # 2. The Generation Loop
        while True:
            # Randomize the order of tasks (e.g. Tuft -> PT -> Cap vs PT -> Cap -> Tuft)
            # This prevents the model from learning a fixed order.
            random.shuffle(self.tasks)

            batches_yielded_this_round = 0

            for t in self.tasks:
                start = pointers[t]

                # Check if we have enough data left in this bucket
                # If we ran out, we re-shuffle and loop back (Oversampling logic)
                # allowing rare classes to be seen as often as common classes.
                if start + self.batch_size > len(self.class_indices[t]):
                    random.shuffle(self.class_indices[t])
                    pointers[t] = 0
                    start = 0

                # Yield the batch indices
                end = start + self.batch_size
                yield self.class_indices[t][start:end]

                # Advance pointer
                pointers[t] += self.batch_size
                batches_yielded_this_round += 1

            # Stop condition: In standard PyTorch, __len__ controls the progress bar,
            # but we break here just in case.
            # (Ideally, the DataLoader stops calling next() when it hits __len__)
            if batches_yielded_this_round == 0:
                break

    def __len__(self):
        return self.num_batches
