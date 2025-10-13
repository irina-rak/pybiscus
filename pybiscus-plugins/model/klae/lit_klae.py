from typing import List, Literal, TypedDict, Union

import lightning.pytorch as pl
import torch

from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.layers import Act
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.utils import set_determinism
from pydantic import BaseModel, ConfigDict
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss

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


class ConfigKLAutoEncoder(BaseModel):
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
    channels: list[int]
    attention_levels: list[int]
    latent_channels: int
    num_res_blocks: int
    norm_num_groups: int

    model_config = ConfigDict(extra="forbid")


class ConfigAlexDiscriminator(BaseModel):
    """A Pydantic Model to validate the discriminator config given by the user.

    Attributes
    ----------
    spatial_dims: int
        number of spatial dimensions
    num_layers_d: int
        number of layers in the discriminator
    channels: int
        number of channels in the discriminator
    in_channels: int
        number of input channels
    out_channels: int
        number of output channels
    """

    spatial_dims: int = 3
    num_layers_d: int = 4
    channels: int = 64
    in_channels: int
    out_channels: int
    kernel_size: int

    model_config = ConfigDict(extra="forbid")


class ConfigKLAE(BaseModel):
    """A Pydantic Model to validate the LitPBR config given by the user.

    Attributes
    ----------
    generator_config: ConfigKLAutoEncoder
        Configuration for the autoencoder
    discriminator_config: ConfigAlexDiscriminator  
        Configuration for the discriminator
    lr: List[float]
        Learning rates for generator and discriminator
    perceptual_weight: float
        Weight for perceptual loss
    adversarial_weight: float
        Weight for adversarial loss
    kl_weight: float
        Weight for KL divergence loss
    adv_weight: float
        Weight for adversarial loss in training
    autoencoder_warm_up_n_epochs: int
        Number of epochs to train only the autoencoder before adding discriminator
    """

    generator_config: ConfigKLAutoEncoder
    discriminator_config: ConfigAlexDiscriminator
    lr: List[float] = [1e-4, 1e-5]
    perceptual_weight: float = 0.001
    kl_weight: float = 1e-6
    reconstruction_weight: float = 1.0  # Increase for better reconstruction quality
    adv_weight: float = 0.01
    autoencoder_warm_up_n_epochs: int = 10
    seed: Union[int, None] = None
    _logging: bool = True

    model_config = ConfigDict(extra="forbid")


class ConfigModel_KLAE(BaseModel):
    """Pydantic BaseModel to validate Configuration for KLAE model.

    Attributes
    ----------
    pretrained_weights_path: Union[str, None]
        Path to the pretrained weights, if any
    name: Literal["klae"]
        designation "klae" to choose
    config:
        configuration for the model LitUnet
    """

    name: Literal["klae"]
    pretrained_weights_path: Union[str, None] = None
    config: ConfigKLAE

    model_config = ConfigDict(extra="forbid")


class KLAESignature(TypedDict):
    """A TypedDict to represent the signature of both training and validation steps of LDM model.

    Used in particular in train and test loops, to gather information on how many metrics are returned by the model.

    Attributes
    ----------
    loss: torch.Tensor
        The main loss (reconstruction loss)
    recons_loss: torch.Tensor
        Reconstruction loss
    kl_loss: torch.Tensor
        KL divergence loss (training only)
    perceptual_loss: torch.Tensor
        Perceptual loss (training only)
    gen_loss: torch.Tensor
        Generator adversarial loss (training only)
    disc_loss: torch.Tensor
        Discriminator loss (training only)
    """

    loss: torch.Tensor
    recons_loss: torch.Tensor
    kl_loss: torch.Tensor
    perceptual_loss: torch.Tensor
    gen_loss: torch.Tensor
    disc_loss: torch.Tensor


class LitKLAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        generator_config: ConfigKLAutoEncoder,
        discriminator_config: ConfigAlexDiscriminator,
        lr: List[float] = [1e-4, 1e-5],
        perceptual_weight: float = 0.001,
        kl_weight: float = 1e-6,
        reconstruction_weight: float = 1.0,
        adv_weight: float = 0.01,
        autoencoder_warm_up_n_epochs: int = 10,
        seed: Union[int, None] = None,
        _logging: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        if seed is not None and type(seed) is int:
            set_determinism(seed=seed)
            console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

        # Disable automatic optimization for manual control
        self.automatic_optimization = False

        # Initialize the generator
        self.autoencoderkl = AutoencoderKL(
            **(generator_config.model_dump() if hasattr(generator_config, 'model_dump') else generator_config)
        )

        # Initialize the discriminator
        self.discriminator = PatchDiscriminator(
            **(discriminator_config.model_dump() if hasattr(discriminator_config, 'model_dump') else discriminator_config)
        )
        
        # Loss functions
        self.l1_loss = L1Loss()

        self.adversarial_loss = PatchAdversarialLoss(
            criterion="least_squares",
        )

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=generator_config.spatial_dims if hasattr(generator_config, 'spatial_dims') else generator_config['spatial_dims'],
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
        )

        # Set weights
        self.reconstruction_weight = reconstruction_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight

        self.autoencoder_warm_up_n_epochs = autoencoder_warm_up_n_epochs
        
        self.lr = lr
        self._logging = _logging
        self._signature = KLAESignature

        # # Get scalers for mixed precision training
        # self.scaler_g, self.scaler_d = self.get_scalers()

    @property
    def signature(self):
        return self._signature

    def forward(self,):
        pass

    def training_step(self, batch: torch.Tensor, batch_idx) -> KLAESignature:
        images = batch["image"]

        # Get optimizers
        opt_g, opt_d = self.optimizers()
        
        # Generator training (AutoEncoder)
        opt_g.zero_grad(set_to_none=True)  # Set to None for memory efficiency

        # scheduler_g, scheduler_d = self.lr_schedulers()

        # with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
        reconstruction, z_mu, z_sigma = self.autoencoderkl(images)
        kl_loss = self.kl_divergence_loss(z_mu, z_sigma)

        recons_loss = self.l1_loss(reconstruction.float(), images.float())
        p_loss = self.perceptual_loss(reconstruction.float(), images.float())
        # Calculate total generator loss
        loss_g = (self.reconstruction_weight * recons_loss) + (self.kl_weight * kl_loss) + (self.perceptual_weight * p_loss)

        # Warm-up phase
        adversarial_loss = torch.tensor(0.0, device=self.device)
        if self.current_epoch > self.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            adversarial_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.adv_weight * adversarial_loss
            
        self.manual_backward(loss_g)
        # torch.nn.utils.clip_grad_norm_(self.autoencoderkl.parameters(), max_norm=1.0)
        opt_g.step()

        discriminator_loss = torch.tensor(0.0, device=self.device)
        if self.current_epoch > self.autoencoder_warm_up_n_epochs:
            # Discriminator training
            opt_d.zero_grad(set_to_none=True)

            # with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)

            logits_real = self.discriminator(images.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)

            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = self.adv_weight * discriminator_loss

            self.manual_backward(loss_d)
            # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            opt_d.step()
        
        if self._logging:
            self.log("train_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
            self.log("train_gen_loss", adversarial_loss, prog_bar=True, sync_dist=True)
            self.log("train_disc_loss", discriminator_loss, prog_bar=True, sync_dist=True)

        return {
            "loss": recons_loss.detach(),
            "recons_loss": recons_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "perceptual_loss": p_loss.detach(),
        }

    def validation_step(self, batch: torch.Tensor, batch_idx) -> KLAESignature:
        with torch.no_grad():
            images = batch["image"]
            
            reconstruction, _, _ = self.autoencoderkl(images)
            recons_loss = self.l1_loss(reconstruction.float(), images.float())
                
        if self._logging:
            self.log("val_recons_loss", recons_loss, prog_bar=True, sync_dist=True)

        return {
            "loss": recons_loss.detach(),
            "recons_loss": recons_loss.detach(),
        }

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]

        outputs = sliding_window_inference(
            images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=self.model
        )
        labels = manual_one_hot(labels.squeeze(1).to(torch.int64), num_classes=outputs.shape[1])
        labels = torch.permute(labels, (0, 4, 1, 2, 3)).float()
        loss = self.loss(outputs, labels)
        return loss

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure the optimizers for the generator and discriminator.

        Returns
        -------
        list[torch.optim.Optimizer]
            A list containing the optimizers for the generator and discriminator.
        """
        opt_gen = torch.optim.AdamW(
            self.autoencoderkl.parameters(), lr=self.lr[0], betas=(0.9, 0.999)
        )
        opt_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.lr[1], betas=(0.9, 0.999)
        )

        # Use CosineAnnealingWarmRestarts for generator
        # This scheduler resets the learning rate periodically, which can help with convergence
        # and exploration of the loss landscape.
        # It allows the learning rate to oscillate, which can help escape local minima.
        # This is particularly useful in GAN training where the generator and discriminator
        # can benefit from periodic learning rate resets.
        # The parameters T_0 and T_mult control the cycle length and how it increases over
        # subsequent cycles.
        # T_0=50 means every 50 epochs, LR resets to initial value
        # T_mult=2 means subsequent cycles are 100, 200, 400 epochs
        scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_gen, T_0=50, T_mult=2, eta_min=1e-6
        )
        scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_disc,
            mode="min", 
            factor=0.7,        # Less aggressive reduction
            patience=10,        # Slightly higher patience for discriminator
            min_lr=5e-7,      # Higher minimum LR for discriminator
            threshold=1e-4,   # Add threshold
            verbose=True      # Log when LR changes
        )
        # return [opt_gen, opt_disc]
        return [
            {
                "optimizer": opt_gen,
                "lr_scheduler": {
                    "scheduler": scheduler_gen,
                    "interval": "epoch",
                    "frequency": 1,
                    # No "monitor" for CosineAnnealingWarmRestarts
                }
            },
            {
                "optimizer": opt_disc,
                "lr_scheduler": {
                    "scheduler": scheduler_disc,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_recons_loss",
                }
            }
        ]
    
    def on_train_epoch_end(self) -> None:
        scheduler_g, scheduler_d = self.lr_schedulers()
        # Step the schedulers
        if self.lr_schedulers() is not None:
            scheduler_g.step()
            # Use the reconstruction loss as a metric for the discriminator scheduler
            r_loss = self.trainer.callback_metrics.get("val_recons_loss", None)
            if r_loss is not None:
                scheduler_d.step(metrics=r_loss)
            else:
                console.log("[bold][red]Warning: 'val_recons_loss' not found in callback metrics.[/red][/bold]")

        if self._logging:
            # self.log("lr_scheduler_g", scheduler_g.get_last_lr()[0], prog_bar=True, sync_dist=True)
            # self.log("lr_scheduler_d", scheduler_d.get_last_lr()[0], prog_bar=True, sync_dist=True)
            self.log("lr_g", self.optimizers()[0].param_groups[0]["lr"], prog_bar=True, sync_dist=True)
            self.log("lr_d", self.optimizers()[1].param_groups[0]["lr"], prog_bar=True, sync_dist=True)

    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate the KL divergence loss.

        Args:
        -----
            mu [torch.Tensor]
                The mean tensor.
            logvar [torch.Tensor]
                The log variance tensor.

        Returns:
        --------
            torch.Tensor
                The KL divergence loss.
        """
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.pow(2) - torch.log(logvar.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.mean(kl_loss)

    
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