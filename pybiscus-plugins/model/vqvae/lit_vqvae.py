from typing import List, Literal, Sequence, Tuple, TypedDict, Union

import lightning.pytorch as pl
import torch

from monai.metrics import FIDMetric
from monai.networks.nets import VQVAE
from monai.utils import set_determinism
from pydantic import BaseModel, ConfigDict
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss

from pybiscus.core.pybiscus_logger import console



class ConfigVQAutoEncoder(BaseModel):
    """A Pydantic Model to validate the generator config given by the user.

    Attributes
    ----------
    in_channels: int
        number of channels of the input
    out_channels: int
        number of channels of the output
    lr: float
        the learning rate
    """

    spatial_dims: int
    in_channels: int
    out_channels: int
    channels: Sequence[int]
    num_res_layers: int
    num_res_channels: Union[Sequence[int], int]
    # downsample_parameters: Union[Sequence[List[int, int, int, int]], List[int, int, int, int]]
    # upsample_parameters: Union[Sequence[List[int, int, int, int]], List[int, int, int, int]]
    num_embeddings: int
    embedding_dim: int
    use_checkpointing: bool = False

    model_config = ConfigDict(extra="forbid")


class ConfigVQVAE(BaseModel):
    """A Pydantic Model to validate the LitPBR config given by the user.

    Attributes
    ----------
    vqvae_config: ConfigVQAutoEncoder
        Configuration for the autoencoder
    lr: List[float]
        Learning rates for generator and discriminator
    seed: Union[int, None]
        Seed for reproducibility, if None, no seed is set
    _logging: bool
        Whether to log training and validation metrics
    """

    vqvae_config: ConfigVQAutoEncoder
    lr: float = 1e-4

    seed: Union[int, None] = None
    _logging: bool = True

    model_config = ConfigDict(extra="forbid")


class ConfigModel_VQVAE(BaseModel):
    """Pydantic BaseModel to validate Configuration for VQVAE model.

    Attributes
    ----------
    pretrained_weights_path: Union[str, None]
        Path to the pretrained weights, if any
    name: Literal["vqvae"]
        designation "vqvae" to choose
    config:
        configuration for the model LitUnet
    """

    name: Literal["vqvae"]
    pretrained_weights_path: Union[str, None] = None
    config: ConfigVQVAE

    model_config = ConfigDict(extra="forbid")


class VQVAE_Signature(TypedDict):
    """A TypedDict to represent the signature of both training and validation steps of LDM model.

    Used in particular in train and test loops, to gather information on how many metrics are returned by the model.

    Attributes
    ----------
    recons_loss: torch.Tensor
        Reconstruction loss, typically L1 loss between input and output images
    quantization_loss: torch.Tensor
        Quantization loss, which measures the difference between the input and the quantized output
    loss: torch.Tensor
        Combined loss, which is the sum of the reconstruction and the quantization losses
    # fid_score: torch.Tensor
    #     Frechet Inception Distance (FID) calculates the distance between the distribution of generated images and real images.
    """

    recons_loss: torch.Tensor
    quantization_loss: torch.Tensor
    loss: torch.Tensor
    # fid_score: torch.Tensor


class LitVQVAE(pl.LightningModule):
    def __init__(
        self,
        vqvae_config: ConfigVQAutoEncoder,
        lr: float = 1e-4,
        seed: Union[int, None] = None,
        _logging: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        if seed is not None and type(seed) is int:
            set_determinism(seed=seed)
            console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

        # Initialize the generator
        self.vqvae = VQVAE(
            **(vqvae_config.model_dump() if hasattr(vqvae_config, "model_dump") else vqvae_config)
        )
        
        # Loss functions
        self.l1_loss = L1Loss()
        
        self.lr = lr
        self._logging = _logging
        self._signature = VQVAE_Signature

    @property
    def signature(self):
        return self._signature

    def forward(self,):
        pass

    def training_step(self, batch: torch.Tensor, batch_idx) -> VQVAE_Signature:
        images = batch["image"]

        scheduler = self.lr_schedulers()

        reconstruction, quantization_loss = self.vqvae(images)
        recons_loss = self.l1_loss(reconstruction.float(), images.float())

        loss = recons_loss + quantization_loss
        
        if self._logging:
            self.log("train_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
            self.log("train_quantization_loss", quantization_loss, prog_bar=True, sync_dist=True)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            if scheduler is not None:
                self.log("lr", scheduler.get_last_lr()[0], prog_bar=True, sync_dist=True)

        return {
            "recons_loss": recons_loss,
            "quantization_loss": quantization_loss,
            "loss": loss,
        }

    def validation_step(self, batch: torch.Tensor, batch_idx) -> VQVAE_Signature:
        with torch.no_grad():
            images = batch["image"]
            
            reconstruction, quantization_loss = self.vqvae(images)
            recons_loss = self.l1_loss(reconstruction.float(), images.float())

            loss = recons_loss + quantization_loss

        if self._logging:
            self.log("val_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
            self.log("val_quantization_loss", quantization_loss, prog_bar=True, sync_dist=True)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return {
            "recons_loss": recons_loss,
            "quantization_loss": quantization_loss,
            "loss": loss,
        }

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimizer and scheduler for the VQ-VAE.

        Returns
        -------
        tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]
            A tuple containing the optimizer and scheduler for the VQ-VAE.
        """
        optimizer = torch.optim.Adam(self.vqvae.parameters(), lr=self.lr)

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr,
        #     total_steps=self.trainer.estimated_stepping_batches,
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            }
        }

    
    # def get_scalers(self) -> tuple[GradScaler, GradScaler]:
    #     """Get the gradient scalers for mixed precision training.

    #     Returns
    #     -------
    #     tuple[GradScaler, GradScaler]
    #         A tuple containing the scalers for the generator and discriminator.
    #     """
    #     scaler_gen = GradScaler(enabled=self.trainer.precision == 16)
    #     scaler_disc = GradScaler(enabled=self.trainer.precision == 16)
    #     return scaler_gen, scaler_disc