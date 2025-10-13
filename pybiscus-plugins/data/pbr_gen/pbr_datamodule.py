from random import random
from typing import Literal, List, Optional

import torch
import lightning.pytorch as pl

from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Orientationd,
    Spacingd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CenterSpatialCropd,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ToTensord,
    Lambdad,
    EnsureTyped,
    Resized,
    RandRotate90d,
    RandAffined,
    Rand3DElasticd,
    RandGaussianNoised
)
from torch.utils.data._utils.collate import default_collate
from pydantic import BaseModel, ConfigDict

from pybiscus.core.pybiscus_logger import console
from pbr_gen.pbr_dataset import CTCacheDataset


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
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-250,
            a_max=600,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=margin),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False)
    ]

    # Training-specific transforms
    train_transforms = Compose(
        common_preprocessing
        + [
            RandCropByPosNegLabeld(
                keys=["image"],
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
                keys=["image"],
                spatial_size=patch_size,
                method="symmetric",
                mode=("edge", "constant")
            ),
            RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
            # RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ToTensord(keys=["image"]),
        ]
    )

    # Validation-specific transforms
    val_transforms = Compose(
        common_preprocessing
    )

    return train_transforms, val_transforms


def get_transforms_ldm(
        patch_size: tuple[int, int, int] = (96, 96, 96),
        pixdim: tuple[float, float, float] = (1.0, 1.0, 2.0),
        margin: int = 25,
):
    """Get the transforms optimized for Latent Diffusion Model training.
    
    Key differences from segmentation transforms:
    - No label-dependent cropping
    - Scale to [-1, 1] range (better for diffusion models)
    - Center cropping for consistent spatial dimensions
    - Focus on image-only preprocessing
    
    Args:
        patch_size (tuple[int, int, int], optional): The size of the patches. Defaults to (96, 96, 96).
        pixdim (tuple[float, float, float], optional): The pixel dimensions. Defaults to (1.0, 1.0, 2.0).
        margin (int, optional): Margin for foreground cropping. Defaults to 25.

    Returns:
        tuple[Compose, Compose]: The training and validation transforms.
    """
    # Common preprocessing for LDM
    # common_preprocessing = [
    #     LoadImaged(keys=["image"]),
    #     EnsureChannelFirstd(keys=["image"]),
    #     Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
    #     EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    #     EnsureTyped(keys=["image"]),
    #     Orientationd(keys=["image"], axcodes="RAS"),
    #     Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
    #     # Choose one: either RandSpatialCropd OR ResizeWithPadOrCropd
    #     # RandSpatialCropd(
    #     #     keys=["image"],
    #     #     roi_size=patch_size,
    #     #     random_center=True,
    #     #     random_size=False
    #     # ),
    #     CenterSpatialCropd(keys=["image"], roi_size=patch_size),
    #     # Resized(keys=["image"], spatial_size=patch_size, mode="trilinear"),  # Resize whole image
    #     # ScaleIntensityRanged(
    #     #     keys=["image"],
    #     #     a_min=-250,
    #     #     a_max=600,
    #     #     b_min=-1.0,  # Scale to [-1, 1] for diffusion models
    #     #     b_max=1.0,
    #     #     clip=True,
    #     # ),
    #     ScaleIntensityRangePercentilesd(
    #         keys=["image"],
    #         lower=0,
    #         upper=99.5,
    #         b_min=0,
    #         b_max=1
    #     ),
    #     # Lambdad(keys=["image"], func=lambda x: torch.clamp(x, 0, 1)),
    # ]
    common_preprocessing = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["image"], func=lambda x: x[0, :, :, :]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
        CropForegroundd(keys=["image"], source_key="image", margin=margin),

        # # This ensures consistent sizes
        # ResizeWithPadOrCropd(
        #     keys=["image"],
        #     spatial_size=patch_size,
        #     mode="constant",
        #     method="symmetric",
        # ),

        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=patch_size,
            num_samples=2,  # 2 patches per image = ~400 total samples
            random_size=False
        ),

        # CenterSpatialCropd(keys=["image"], roi_size=patch_size),
        SpatialPadd(
            keys=["image"],
            spatial_size=patch_size,
            method="symmetric",
            mode=("constant")
        ),

        # RandSpatialCropd( # Use this with ResizeWithPadOrCropd when setting batch size > 1, otherwise a size mismatch occurs
        #     keys=["image"],
        #     roi_size=patch_size,
        #     random_size=False
        # ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0,
            upper=99.5,
            b_min=0,
            b_max=1
        ),
    ]

    # Training-specific transforms for LDM
    train_transforms = Compose(
        common_preprocessing
        + [
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.3),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.3),
            # RandRotate90d(keys=["image"], prob=0.1, max_k=1),
            RandAffined(
                keys=["image"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear")
            ),
            Rand3DElasticd(
                keys=["image"],
                sigma_range=(5, 8),
                magnitude_range=(50, 150),
                prob=0.2,
                mode=("bilinear"),
                spatial_size=patch_size,
                padding_mode="zeros"
            ),
            # RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.005),
            # RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.2),
            # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
        # + [
        #     # RandSpatialCropd(keys=["image"], roi_size=patch_size, random_center=True, random_size=False),
        #     CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 64)),
        #     # Geometric augmentations
        #     RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
        #     RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
        #     # RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
        #     # Intensity augmentations (lighter for diffusion models)
        #     RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.3),
        #     RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.3),
        # ]
    )

    # Validation transforms (no augmentation, use center crop for consistency)
    val_preprocessing = common_preprocessing.copy()
    # Replace RandSpatialCropd with CenterSpatialCropd for validation
    for i, transform in enumerate(val_preprocessing):
        if isinstance(transform, RandSpatialCropd):
            val_preprocessing[i] = CenterSpatialCropd(keys=["image"], roi_size=patch_size)
            break
    
    val_transforms = Compose(val_preprocessing)

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
    augment: bool, optional
        whether to use augmentation of data (default to False)
    preprocessed: bool, optional
        whether the data have already been preprocessed or not (default to True)
    cache_rate: float, optional
        cache rate for the dataset (default to 1.0)
    num_workers: int, optional
        the number of workers for the DataLoaders (default to 0)
    task_type: str, optional
        the type of task: "segmentation" or "generation" (default to "segmentation")
    """

    dir_train: str
    dir_val: str
    dir_test: str = None
    batch_size: int = 1
    # shape_img: List = [96, 96, 96]
    patch_size: List[int] = [96, 96, 96]
    spacing: List[float] = [1.0, 1.0, 2.0]
    margin: int = 25
    augment: bool = False
    preprocessed: bool = True
    cache_rate: float = 1.0
    num_workers: int = 0
    task_type: str

    model_config = ConfigDict(extra="forbid")


class ConfigData_PBR(BaseModel):
    name: Literal["pbr_gen"]
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
        patch_size: tuple[float, float, float, float] = (96, 96, 96),
        spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
        margin: int = 25,
        augment: bool = False,
        preprocessed: bool = True,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        task_type: str = "segmentation",  # "segmentation" or "generation"
    ):
        super().__init__()
        self.data_dir_train = dir_train
        self.data_dir_val = dir_val
        self.data_dir_test = dir_test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.augment = augment
        self.cache_rate = cache_rate
        self.task_type = task_type

        # Choose transforms based on task type
        # For generative modeling (LDM), use specialized transforms
        if hasattr(self, "task_type") and self.task_type == "generation":
            self.train_transforms, self.val_transforms = get_transforms_ldm(
                patch_size=tuple(patch_size),
                pixdim=spacing,
                margin=margin,
            )
        else:
            # Default segmentation transforms
            self.train_transforms, self.val_transforms = get_transforms(
                patch_size=tuple(patch_size),
                pixdim=spacing,
                margin=margin,
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

            # self.data_train = DecathlonDataset(
            #     root_dir=self.data_dir_train,
            #     task="Task01_BrainTumour",
            #     section="training",
            #     cache_rate=self.cache_rate,
            #     num_workers=self.num_workers,
            #     seed=42,  # Ensure reproducibility
            #     transform=self.train_transforms,
            # )
            # self.data_val = DecathlonDataset(
            #     root_dir=self.data_dir_val,
            #     task="Task01_BrainTumour",
            #     section="validation",
            #     cache_rate=self.cache_rate,
            #     num_workers=self.num_workers,
            #     seed=42,  # Ensure reproducibility
            #     transform=self.val_transforms,
            # )

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
        collated = {"image": torch.stack([item["image"] for item in batch])}
        
        # Only add labels if they exist (for segmentation tasks)
        if "label" in batch[0]:
            collated["label"] = torch.stack([item["label"] for item in batch])
            
        return collated
    
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
            persistent_workers=True,  # Keep workers alive for faster training
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
            persistent_workers=True,
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
