import json

from glob import glob
from os import path, listdir
from pathlib import Path

from monai.data import CacheDataset, Dataset
from monai.transforms import Compose

from pybiscus.core.pybiscus_logger import console



class CTCacheDataset:
    def __init__(
        self,
        data_dir: str,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        transforms: Compose = None
    ):
        self.data_dir = Path(data_dir)
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.transforms = transforms

        # Prepare data list
        self.data = self.create_data_list()

        # Create CacheDataset
        self.dataset = CacheDataset(
            data=self.data,
            transform=self.transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
            # copy_cache=False,
            # runtime_cache="processes",
        )

    def __len__(self):
        return len(self.dataset)

    def create_data_list(self):
        if self.data_dir.is_file():
            console.log(f"Loading data from {self.data_dir}")
            with open(self.data_dir, "r") as f:
                return json.load(f)
        
        console.log(f"Loading data from directory {self.data_dir}")
        cases = [d for d in listdir(self.data_dir) if path.isdir(self.data_dir / d)]

        images = []
        # labels = []
        names = []
        for case in cases:
            image_path = str(self.data_dir / case / "CT" / "image.nii.gz")
            # label_path = str(self.data_dir / case / "Labels" / "combined_labels.nii.gz")
            if path.exists(image_path):
                images.append(image_path)
            # if path.exists(label_path):
            #     labels.append(label_path)
            names.append(case)
            
        # data = [{"image": image_name, "label": label_name, "name": name} for image_name, label_name, name in zip(images, labels, names)]
        data = [{"image": image_name, "name": name} for image_name, name in zip(images, names)]
        return data

    def get_dataset(self):
        return self.dataset
