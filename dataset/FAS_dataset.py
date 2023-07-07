import sys
from typing import Iterable, Union
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from collections import defaultdict, Counter

sys.path.append(Path(__file__).resolve().parent.as_posix())
from turbojpeg_singleton import jpeg_reader


class FASDataset(Dataset):

    def __init__(self, root_dir: Path, csv_path: Path, transform=None, smoothing: bool = True, is_train: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_path.as_posix())
        self.transform = transform
        self.is_train = is_train
        self.name = self.path_to_name(root_dir)
        
        if smoothing:  # Not really used yet
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

    @staticmethod
    def path_to_name(root_dir: Union[Path, str]) -> str:
        root_dir = Path(root_dir)
        return f"{root_dir.parent.name}_{root_dir.name}" 

    def __getitem__(self, index):
        img_name = Path(self.root_dir, self.data.iloc[index, 0])
        label = self.data.iloc[index, 1]
        # img_name = os.path.join(self.root_dir, "images", img_name)
        
        if img_name.suffix in (".jpg", ".jpeg"):
            with open(img_name, "rb") as f:
                img = jpeg_reader.decode(f.read())
        else:
            img = cv2.imread(img_name.as_posix()) 
        label = label.astype(np.float32)
        label = np.expand_dims(label, axis=0)
        
        if self.transform:
            img1 = self.transform(image=img)
            if self.is_train:
                img2 = self.transform(image=img)
                return img1["image"], img2["image"], label
        return img1["image"], label

    def __len__(self):
        return len(self.data)
    
    # def get_class_counts(self) -> Counter:
    #     targets = self.data[1].tolist()
    #     class_counts = Counter(targets)
    #     return class_counts
    
    # def get_class_indexes(self) -> Dict[int, List[int]]:
    #     indexes_dict = defaultdict(list)
    #     for i, original_index in enumerate(indexes):
    #         class_idx = self.targets_all[original_index]
    #         indexes_dict[class_idx].append(i)
    #     return indexes_dict



class ConcatBinaryClassificationDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

        self.__compute_class_counts()
        self.__compute_class_indices()


    def __compute_class_counts(self) -> None:
        self.class_counts = Counter({0: 0, 1: 0})
        for ds in self.datasets:
            self.class_counts += ds.get_class_counts()

    def __compute_class_indices(self) -> None:
        bias = 0
        self.class_indices = defaultdict(list)
        for ds in enumerate(self.datasets):
            current_class_indexes = ds.get_class_indexes()
            for k, v in current_class_indexes.items():
                shifted_indexes = [value + bias for value in v]
                self.class_indices[k] += shifted_indexes
            bias += len(ds)   
