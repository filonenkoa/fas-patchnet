import sys
from typing import Iterable, List, Union
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
        self.data = pd.read_csv(csv_path.as_posix(), names=["path", "label"])
        self.transform = transform
        self.is_train = is_train
        self.name = self.path_to_name(root_dir)
        self.labels = self.data["label"].tolist()
        
        if smoothing:  # Not really used yet
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99
            
        self.return_images = True

    @staticmethod
    def path_to_name(root_dir: Union[Path, str]) -> str:
        root_dir = Path(root_dir)
        return f"{root_dir.parent.name}_{root_dir.name}"

    def __getitem__(self, index):
        # label = self.data.iloc[index, 1]
        label = self.labels[index]
        label = np.array([label], dtype=np.float32)
        label = np.expand_dims(label, axis=0)
        
        if self.return_images:
            img_name = Path(self.root_dir, self.data.iloc[index, 0])
            if img_name.suffix in (".jpg", ".jpeg"):
                with open(img_name, "rb") as f:
                    img = jpeg_reader.decode(f.read())
            else:
                img = cv2.imread(img_name.as_posix()) 

            if self.transform:
                img1 = self.transform(image=img)
                if self.is_train:
                    img2 = self.transform(image=img)
                    return img1["image"], img2["image"], label
            return img1["image"], label
        else:
            return None, label
    

    def __len__(self):
        return len(self.data)


class SampleDataset(Dataset):
    """
    A dummy dataset to test a sampler
    """
    def __init__(self, class_ratio: float = 4.0/1.0, base_number: int = 100):
        super().__init__()
        self.labels = np.array([0] * int(base_number * class_ratio) + [1] * base_number, dtype=np.int64)
        # self.values = [100] * len(self.labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index) -> int:
        return 111, self.labels[index]
    
    def __repr__(self):
        classes_num = Counter(self.labels)
        return f"Number of classes {classes_num[0]} and {classes_num[1]}"


class ConcatDatasetWithLabels(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        self.labels = []
        for dataset in self.datasets:
            self.labels.extend(dataset.labels)
