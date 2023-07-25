import math
from torch.utils.data import Dataset, DistributedSampler
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader

    
class DistributedBalancedSampler(DistributedSampler):
    """
    A PyTorch DistributedSampler that supports oversampling for imbalanced classes. if `shuffle` is False,
    then works as DistributedSampler without balancing.

    Args:
        dataset (Dataset): The dataset to sample from.
        num_replicas (Optional[int]): The number of replicas to distribute data across. Defaults to None.
        rank (Optional[int]): The rank of the current process within num_replicas. Defaults to None.
        shuffle (bool): Whether to shuffle the indices before returning them. Defaults to True.
        seed (int): The random seed to use when shuffling indices. Defaults to 0.
    """
    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False):
        super().__init__(dataset,
                         num_replicas=num_replicas,
                         rank=rank,
                         shuffle=shuffle,
                         seed=seed,
                         drop_last=drop_last)
        
        self.dataset = dataset
        self.class_indices = self._get_class_indices(dataset)
        self.class_counts = self._get_class_counts()
        self.max_class_count = max(self.class_counts.values())
        self.total_samples_num = self._compute_total_num_samples()
        

    def _get_class_counts(self):
        return {class_label: len(indices) for class_label, indices in self.class_indices.items()}
    
    def _compute_total_num_samples(self) -> int:
        total_samples = 0
        for k in self.class_indices.keys():
            total_samples += self.class_counts[k] * (self.max_class_count // self.class_counts[k])
        return total_samples

    @staticmethod
    def _get_class_indices(dataset: Dataset) -> Dict[int, int]:
        class_indices = defaultdict(list)
        for idx, data in enumerate(dataset):
            class_indices[data[-1].item()].append(idx)
        return class_indices

    def __iter__(self):
        # Shuffle
        if self.shuffle:
            # Enlarge the indices so random can get the values with almost even propability
            inflated_dataset_indices = []
            for k in self.class_indices.keys():
                inflated_dataset_indices += self.class_indices[k] * (self.max_class_count // self.class_counts[k])
            g = np.random.Generator(np.random.PCG64(seed=self.seed + self.epoch))
            indices = g.choice(a=inflated_dataset_indices,
                    size=self.total_size,
                    replace=False).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_samples_num - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            assert len(indices) == self.total_samples_num, f"{len(indices)} vs {self.total_samples_num}"
            # subsample
            indices = indices[self.rank:self.total_samples_num:self.num_replicas]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
            assert len(indices) == self.total_size, f"{len(indices)} vs {self.total_samples_num}"
            indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)

    def __len__(self):
        return self.total_samples_num
    

if __name__ == "__main__":
    from dataset.FAS_dataset import SampleDataset
    dataset = SampleDataset()
    print(dataset)
    sampler = DistributedBalancedSampler(dataset=dataset,
                                         num_replicas=2,
                                         rank=1,
                                         shuffle=True,
                                         seed=1,
                                         drop_last=False)
    loader = DataLoader(dataset=dataset,
                        batch_size=45,
                        sampler=sampler)
    
    print(f"Num batches = {len(loader)}")
    total = Counter()
    for batch_data in loader:
        current_counter = Counter(batch_data[1].tolist())
        total.update(current_counter)
        print(current_counter)
    print(total)