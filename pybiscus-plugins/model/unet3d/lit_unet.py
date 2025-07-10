from typing import Literal, TypedDict

import lightning.pytorch as pl
import torch

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceHelper
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    AsDiscrete,
)
from pydantic import BaseModel, ConfigDict

from pybiscus.core.pybiscus_logger import console



def manual_one_hot(
    tensor: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Manually one-hot encode the label tensor in the batch.

    Args:
        batch (dict[str, torch.Tensor]): The input batch containing the label tensor.
        num_classes (int): The number of classes for one-hot encoding.

    Returns:
        dict[str, torch.Tensor]: The updated batch with the one-hot encoded label tensor.
    """
    B, _, D, H, W = tensor.shape

    # Remove channel dimension to get indices of shape [B, D, H, W]
    indices = tensor.squeeze(1).long()

    onehot = torch.zeros(B, num_classes, D, H, W).to(tensor.device)
    # Use scatter to create one-hot encoding
    # The first argument is the dimension along which to scatter (1 for channels)
    # The second argument is the indices of the class labels
    # The third argument is the value to scatter (1 for one-hot encoding)
    onehot.scatter_(1, indices.unsqueeze(1), 1)

    return onehot


class ConfigUnet(BaseModel):
    """A Pydantic Model to validate the LitPBR config given by the user.

    Attributes
    ----------
    in_channels: int
        number of channels of the input
    out_channels: int
        number of channels of the output
    lr: float
        the learning rate
    """

    spatial_dims: int = 3
    in_channels: int
    out_channels: int
    channels: list[int]
    strides: list[int]
    num_res_units: int
    norm: str = "BATCH"
    lr: float

    model_config = ConfigDict(extra="forbid")


class ConfigModel_Unet(BaseModel):
    """Pydantic BaseModel to validate Configuration for "paroma" Model.

    Attributes
    ----------
    name: Literal["unet"]
        designation "unet" to choose
    config:
        configuration for the model LitUnet
    """

    name: Literal["unet"]
    config: ConfigUnet

    model_config = ConfigDict(extra="forbid")


class UnetSignature(TypedDict):
    """A TypedDict to represent the signature of both training and validation steps of Unet model.

    Used in particular in train and test loops, to gather information on how many metrics are returned by the model.

    Attributes
    ----------
    loss: torch.Tensor
    dice_avg: torch.Tensor
    dice_0: torch.Tensor
    dice_1: torch.Tensor
    dice_2: torch.Tensor
    """

    loss: torch.Tensor
    dice_avg: torch.Tensor
    
    


class LitUnet(pl.LightningModule):
    def __init__(
        # self, in_channels: int, out_channels: int, lr: float, _logging: bool = False
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: list[int],
        strides: list[int],
        num_res_units: int,
        norm: str,
        lr: float,
        _logging: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )
        # self.loss = CEDiceLoss(class_weights=None)
        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            smooth_nr=1e-6,
            smooth_dr=1e-6,
        )
        # self.dice_score = DiceScore()
        self.dice_score = DiceHelper(
            include_background=False,
            softmax=True,
            reduction="mean",
            get_not_nans=False,
            num_classes=out_channels,
        )
        self.lr = lr
        self._logging = _logging
        self._signature = UnetSignature

    @property
    def signature(self):
        return self._signature

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch: torch.Tensor, batch_idx) -> UnetSignature:
        images, labels = batch["image"], batch["label"]

        outputs = self.forward(images)

        loss = self.loss(outputs, labels)
        dice = self.dice_score(outputs, labels)
        if self._logging:
            self.log("train_loss", loss, prog_bar=True)
            self.log_dict(
                {f"train_{key}": val for key, val in dice.items()}, prog_bar=True
            )
        results = {"loss": loss}
        
        results["dice_avg"] = dice.item()
        # for key, val in dice.items():
        #     results[key] = val
        return results

    def validation_step(self, batch: torch.Tensor, batch_idx) -> UnetSignature:
        images, labels, name = batch["image"], batch["label"], batch["name"]
        
        outputs = sliding_window_inference(
            images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=self.model
        )
        
        loss = self.loss(outputs, labels)
        dice = self.dice_score(outputs, labels)
        if self._logging:
            self.log("val_loss", loss, prog_bar=True)
            self.log_dict(
                {f"val_{key}": val for key, val in dice.items()}, prog_bar=True
            )
        self.log("val_loss", loss, prog_bar=True)
        results = {"loss": loss}
        
        results["dice_avg"] = dice.item()
        # for key, val in dice.items():
        #     results[key] = val
        return results

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]

        outputs = sliding_window_inference(
            images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=self.model
        )
        labels = manual_one_hot(labels.squeeze(1).to(torch.int64), num_classes=outputs.shape[1])
        labels = torch.permute(labels, (0, 4, 1, 2, 3)).float()
        loss = self.loss(outputs, labels)
        return loss

    def configure_optimizers(self) -> None:
        # return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.99)
        # return bnb.optim.Adam8bit(self.parameters(), lr=self.lr)
        return torch.optim.Adam(self.parameters(), lr=self.lr)
