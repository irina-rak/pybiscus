from typing import List, Literal, Sequence, Tuple, TypedDict, Union

import lightning.pytorch as pl
import torch

from monai.losses import JukeboxLoss, PatchAdversarialLoss, PerceptualLoss
from monai.metrics import FIDMetric
from monai.networks.layers import Act
from monai.networks.nets import PatchDiscriminator, VQVAE
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


class ConfigVQVAEGAN(BaseModel):
    """A Pydantic Model to validate the LitVQVAEGAN config given by the user.

    Attributes
    ----------
    vqvae_config: ConfigVQAutoEncoder
        Configuration for the autoencoder
    discriminator_config: ConfigAlexDiscriminator
        Configuration for the discriminator
    perceptual_weight: float
        Weight for the perceptual loss
    reconstruction_weight: float
        Weight for the reconstruction loss
    adv_weight: float
        Weight for the adversarial loss
    jukebox_weight: float
        Weight for the Jukebox loss
    autoencoder_warm_up_n_epochs: int
        Number of epochs to warm up the autoencoder before using entire GAN training
    lr: List[float]
        Learning rates for generator and discriminator
    seed: Union[int, None]
        Seed for reproducibility, if None, no seed is set
    _logging: bool
        Whether to log training and validation metrics
    """

    vqvae_config: ConfigVQAutoEncoder
    discriminator_config: ConfigAlexDiscriminator
    perceptual_weight: float = 0.001
    reconstruction_weight: float = 1.0  # Increase for better reconstruction quality
    adv_weight: float = 0.01
    jukebox_weight: float = 0.1
    autoencoder_warm_up_n_epochs: int = 10 # Usually n_epochs // 10

    lr: List[float] = [3e-4, 5e-4]  # [generator_lr, discriminator_lr]

    seed: Union[int, None] = None
    _logging: bool = True

    model_config = ConfigDict(extra="forbid")


class ConfigModel_VQVAEGAN(BaseModel):
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

    name: Literal["vqvaegan"]
    pretrained_weights_path: Union[str, None] = None
    config: ConfigVQVAEGAN

    model_config = ConfigDict(extra="forbid")


class VQVAEGAN_Signature(TypedDict):
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


class LitVQVAEGAN(pl.LightningModule):
    def __init__(
        self,
        vqvae_config: ConfigVQAutoEncoder,
        discriminator_config: ConfigAlexDiscriminator,
        perceptual_weight: float = 0.001,
        reconstruction_weight: float = 1.0,  # Increase for better reconstruction quality
        adv_weight: float = 0.01,
        jukebox_weight: float = 1.0,
        autoencoder_warm_up_n_epochs: int = 10,  # Usually n_epochs // 10
        lr: List[float] = [3e-4, 5e-4],
        seed: Union[int, None] = None,
        _logging: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Disable automatic optimization for manual control
        self.automatic_optimization = False

        if seed is not None and type(seed) is int:
            set_determinism(seed=seed)
            console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

        # Initialize the generator
        self.vqvae = VQVAE(
            **(vqvae_config.model_dump() if hasattr(vqvae_config, "model_dump") else vqvae_config)
        )
        
        # Initialize the discriminator
        self.discriminator = PatchDiscriminator(
            **(discriminator_config.model_dump() if hasattr(discriminator_config, 'model_dump') else discriminator_config),
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )
        
        # Loss functions
        self.l1_loss = L1Loss()

        self.adversarial_loss = PatchAdversarialLoss(
            criterion="least_squares",
        )

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=vqvae_config.spatial_dims if hasattr(vqvae_config, 'spatial_dims') else vqvae_config['spatial_dims'],
            network_type="alex",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
        )

        self.jukebox_loss = JukeboxLoss(
            spatial_dims=vqvae_config.spatial_dims if hasattr(vqvae_config, "spatial_dims") else vqvae_config['spatial_dims']
        )

        # Set weights
        self.reconstruction_weight = reconstruction_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.jukebox_weight = jukebox_weight

        self.autoencoder_warm_up_n_epochs = autoencoder_warm_up_n_epochs
        
        self.lr = lr
        self._logging = _logging
        self._signature = VQVAEGAN_Signature

        self.optimizer_gen, self.optimizer_disc = self.configure_optimizers()
        self.scheduler_gen, self.scheduler_disc = self.configure_schedulers()

    @property
    def signature(self):
        return self._signature

    def forward(self,):
        pass

    def training_step(self, batch: torch.Tensor, batch_idx) -> VQVAEGAN_Signature:
        images = batch["image"]

        # Get optimizers
        opt_g, opt_d = self.optimizer_gen, self.optimizer_disc
        
        # Get gradient scalers for mixed precision
        scaler = self.trainer.precision_plugin.scaler if hasattr(self.trainer.precision_plugin, 'scaler') else None

        # Generator training
        opt_g.zero_grad(set_to_none=True)

        # with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
        reconstruction, quantization_loss = self.vqvae(images)

        # Check for NaN values in reconstruction - with recovery
        if torch.isnan(reconstruction).any() or torch.isnan(quantization_loss).any():
            console.log(f"[bold][red]NaN detected in VQVAE output at batch {batch_idx}[/red][/bold]")
            console.log(f"Input stats: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
            console.log(f"Reconstruction NaN count: {torch.isnan(reconstruction).sum()}")
            console.log(f"Quantization loss: {quantization_loss}")

        recons_loss = self.l1_loss(reconstruction.float(), images.float())
        p_loss = self.perceptual_loss(reconstruction.float(), images.float())
        j_loss = self.jukebox_loss(reconstruction.float(), images.float())
        
        loss_g = (
            (self.reconstruction_weight * recons_loss)
            + quantization_loss + (self.perceptual_weight * p_loss)
            + (self.jukebox_weight * j_loss)
        )

        # Warm-up phase
        adversarial_loss = torch.tensor(0.0, device=self.device)
        if self.current_epoch > self.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            adversarial_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.adv_weight * adversarial_loss

        # Scale and backward
        if scaler:
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()
        else:
            self.manual_backward(loss_g)
            opt_g.step()

        # Discriminator training
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
            if self.trainer.precision == 16:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            opt_d.step()
        
        if self._logging:
            self.log("train_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
            self.log("train_quantization_loss", quantization_loss, prog_bar=True, sync_dist=True)
            self.log("train_adv_loss", adversarial_loss, prog_bar=True, sync_dist=True)
            self.log("train_disc_loss", discriminator_loss, prog_bar=True, sync_dist=True)

        return {
            "loss": recons_loss.detach(),
            "recons_loss": recons_loss.detach(),
            "perceptual_loss": p_loss.detach(),
        }

    def validation_step(self, batch: torch.Tensor, batch_idx) -> VQVAEGAN_Signature:
        with torch.no_grad():
            images = batch["image"]
            
            reconstruction, quantization_loss = self.vqvae(images)
            recons_loss = self.l1_loss(reconstruction.float(), images.float())
            p_loss = self.perceptual_loss(reconstruction.float(), images.float())
            j_loss = self.jukebox_loss(reconstruction.float(), images.float())

            loss = (
                (self.reconstruction_weight * recons_loss)
                + quantization_loss + (self.perceptual_weight * p_loss)
                + (self.jukebox_weight * j_loss)
            )

            adversarial_loss = torch.tensor(0.0, device=self.device)
            if self.current_epoch > self.autoencoder_warm_up_n_epochs:
                # with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                adversarial_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)

                loss += self.adv_weight * adversarial_loss

        if self._logging:
            self.log("val_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
            self.log("val_quantization_loss", quantization_loss, prog_bar=True, sync_dist=True)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return {
            "loss": recons_loss.detach(),
            "recons_loss": recons_loss.detach(),
        }

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        """Configure optimizers only - handle schedulers manually."""
        opt_gen = torch.optim.AdamW(
            self.vqvae.parameters(), lr=self.lr[0], betas=(0.9, 0.999)
        )
        opt_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.lr[1], betas=(0.9, 0.999)
        )
        
        return [opt_gen, opt_disc]
    
    def configure_schedulers(self):
        """Configure schedulers for the optimizers."""
        scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gen, mode="min", factor=0.7, patience=10, verbose=True
        )
        scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_disc, mode="min", factor=0.7, patience=10, verbose=True
        )
        
        return [scheduler_gen, scheduler_disc]
    
    def on_before_optimizer_step(self, optimizer):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def on_validation_epoch_end(self) -> None:
        """Handle scheduler stepping after validation."""
        # Get validation loss and step schedulers
        val_loss = self.trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            self.scheduler_gen.step(val_loss)
            self.scheduler_disc.step(val_loss)
            
            if self._logging:
                self.log("lr_g", self.scheduler_gen.optimizer.param_groups[0]["lr"], prog_bar=True, sync_dist=True)
                self.log("lr_d", self.scheduler_disc.optimizer.param_groups[0]["lr"], prog_bar=True, sync_dist=True)

    
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