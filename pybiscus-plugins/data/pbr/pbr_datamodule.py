from typing import Literal, List, Optional

import torch
import lightning.pytorch as pl

from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGridPatchd
)
from torch.utils.data._utils.collate import default_collate
from pydantic import BaseModel, ConfigDict

from pybiscus.core.pybiscus_logger import console
from pbr.pbr_dataset import CTCacheDataset


def get_transforms(
        patch_size: tuple[int, int, int] = (96, 96, 96),
        pixdim: tuple[float, float, float] = (1.0, 1.0, 2.0),
        margin: int = 25,
):
    """Get the transforms for training and validation. The training transforms
    include intensity normalization, patches extraction using RandGridPatchd, and data augmentation.
    The validation transforms include intensity normalization.

    Args:
        patch_size (tuple[int, int, int], optional): The size of the patches to be extracted. Defaults to (96, 96, 96).
        pixdim (tuple[float, float, float], optional): The pixel dimensions. Defaults to (1.0, 1.0, 2.0).

    Returns:
        tuple[Compose, Compose]: The training and validation transforms.
    """
    # Common preprocessing
    common_preprocessing = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-250,
            a_max=600,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=margin),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False)
    ]

    # Training-specific transforms
    train_transforms = Compose(
        common_preprocessing
        + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=patch_size,
                method="symmetric",
                mode=("edge", "constant")
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
    )

    # Validation-specific transforms
    val_transforms = Compose(
        common_preprocessing
    )

    return train_transforms, val_transforms


class ConfigPBR(BaseModel):
    """A Pydantic Model to validate the MedicalLitDataModule config givent by the user.

    Attributes
    ----------
    dir_train: str
        path to the directory holding the training data
    dir_val: str
        path to the directory holding the validating data
    dir_test: str, optional
        path to the directory holding the testing data
    batch_size: int, optional
        the batch size (default to 1)
    shape_img: tuple[float, float, float, float], optional
        the shape of the image (default to (96, 96, 96))
    shape_label: tuple[float, float, float, float], optional
        the shape of the label (default to (96, 96, 96))
    augment: bool, optional
        whether to use augmentation of data (default to False)
    preprocessed: bool, optional
        whether the data have already been preprocessed or not (default to True)
    num_workers: int, optional
        the number of workers for the DataLoaders (default to 0)
    """

    dir_train: str
    dir_val: str
    dir_test: str = None
    batch_size: int = 1
    shape_img: List = [96, 96, 96]
    augment: bool = False
    preprocessed: bool = True
    cache_rate: float = 1.0
    num_workers: int = 0

    model_config = ConfigDict(extra="forbid")


class ConfigData_PBR(BaseModel):
    name: Literal["pbr"]
    config: ConfigPBR

    model_config = ConfigDict(extra="forbid")


class PBRLitDataModule(pl.LightningDataModule):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    pl : _type_
        _description_
    """

    def __init__(
        self,
        # root_dir,
        dir_train: str,
        dir_val: str,
        dir_test: str,
        batch_size: int = 1,
        shape_img: tuple[float, float, float, float] = (96, 96, 96),
        augment: bool = False,
        preprocessed: bool = True,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir_train = dir_train
        self.data_dir_val = dir_val
        self.data_dir_test = dir_test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.augment = augment
        self.cache_rate = cache_rate

        # self.train_transforms = None
        # self.val_transforms = None
        # if preprocessed:
        self.train_transforms, self.val_transforms = get_transforms(
            # patch_size=shape_img,
            # pixdim=(1.0, 1.0, 2.0),
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.data_train = CTCacheDataset(
                data_dir=self.data_dir_train,
                transforms=self.train_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            ).get_dataset()
            self.data_val = CTCacheDataset(
                data_dir=self.data_dir_val,
                transforms=self.val_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            ).get_dataset()

        if stage == "test" or stage is None:
            self.data_test = CTCacheDataset(
                data_dir=self.data_dir_test,
                transforms=self.val_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            ).get_dataset()

    def collate_fn(self, batch):
        """Custom collate function to handle the batch of data.

        Args:
            batch (list): A list of dictionaries containing the data.

        Returns:
            dict: A dictionary containing the collated data.
        """
        # console.log(f"Batch size: {len(batch)}")
        # console.log(f"Batch shape: {batch[0]['image'].shape}")
        # console.log(f"Batch label shape: {batch[0]['label'].shape}")
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
        }
    
    def list_data_collate(self, batch):
        """
        Enhancement for PyTorch DataLoader default collate.
        If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
        Then it's same as the default collate behavior.
        Note:
            Need to use this collate if apply some transforms that can generate batch data.
        """
        elem = batch[0]
        data = torch.stack([i for k in batch for i in k] if isinstance(elem, list) else batch)
        return default_collate(data)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # drop_last=True,
            shuffle=True,
            # collate_fn=self.collate_fn,
            # collate_fn=self.list_data_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # drop_last=True,
            shuffle=False,
            # collate_fn=self.collate_fn,
            # collate_fn=self.list_data_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # drop_last=True,
            shuffle=False,
            # collate_fn=self.collate_fn,
            # collate_fn=self.list_data_collate,
        )
