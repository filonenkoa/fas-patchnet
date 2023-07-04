import os
import sys
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
from pathlib import Path


sys.path.append(Path(__file__).resolve().parent.as_posix())
from turbojpeg_singleton import jpeg_reader

class FASDataset(Dataset):

    def __init__(self, root_dir: Path, csv_path: Path, transform=None, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_path.as_posix())
        self.transform = transform
        
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

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
            img2 = self.transform(image=img)

        return img1["image"], img2["image"], label

    def __len__(self):
        return len(self.data)
