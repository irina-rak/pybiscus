from typing import List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import lightning.pytorch as pl
import torch

from monai.losses import JukeboxLoss, PatchAdversarialLoss, PerceptualLoss
from monai.metrics import FIDMetric
from monai.networks.layers import Act
from monai.networks.nets import PatchDiscriminator, VQVAE
from monai.utils import set_determinism
from pydantic import BaseModel, ConfigDict
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
    manual_accumulate_grad_batches: int = 4  # Manual gradient accumulation steps

    seed: Union[int, None] = None
    compile_model: bool = False  # Whether to compile the model for performance
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
    perceptual_loss: torch.Tensor
        Perceptual loss, which measures the difference between the input and output images in a perceptually meaningful way
    quantization_loss: torch.Tensor
        Quantization loss, which measures the difference between the input and the quantized output
    adversarial_loss: torch.Tensor
        Adversarial loss, which measures how well the generator fools the discriminator
    discriminator_loss: Optional[torch.Tensor]
        Discriminator loss, which measures how well the discriminator distinguishes between real and fake images
    """

    recons_loss: torch.Tensor
    perceptual_loss: torch.Tensor
    quantization_loss: torch.Tensor
    adversarial_loss: torch.Tensor
    discriminator_loss: Optional[torch.Tensor]


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
        manual_accumulate_grad_batches: int = 4,
        seed: Union[int, None] = None,
        compile_model: bool = False,
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
            **(vqvae_config.model_dump() if hasattr(vqvae_config, "model_dump") else vqvae_config),
        )
        
        # Initialize the discriminator
        self.discriminator = PatchDiscriminator(
            **(discriminator_config.model_dump() if hasattr(discriminator_config, "model_dump") else discriminator_config),
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )

        # Compile models if requested
        if compile_model:
            console.log("[bold][blue]Compiling VQVAE and Discriminator for optimized performance...[/blue][/bold]")
            try:
                self.vqvae = torch.compile(self.vqvae, mode="default")
                self.discriminator = torch.compile(self.discriminator, mode="default")
                console.log("[bold][green]Models compiled successfully![/green][/bold]")
            except Exception as e:
                console.log(f"[bold][yellow]Model compilation failed, continuing without: {e}[/yellow][/bold]")
        
        # Loss functions
        self.l1_loss = L1Loss()

        self.adversarial_loss = PatchAdversarialLoss(
            criterion="least_squares",
        )

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=vqvae_config.spatial_dims if hasattr(vqvae_config, "spatial_dims") else vqvae_config["spatial_dims"],
            network_type="alex",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
        )

        self.jukebox_loss = JukeboxLoss(
            spatial_dims=vqvae_config.spatial_dims if hasattr(vqvae_config, "spatial_dims") else vqvae_config["spatial_dims"]
        )

        # Set weights
        self.reconstruction_weight = reconstruction_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.jukebox_weight = jukebox_weight

        self.autoencoder_warm_up_n_epochs = autoencoder_warm_up_n_epochs
        
        self.lr = lr
        self.manual_accumulate_grad_batches = manual_accumulate_grad_batches
        self.compile_model = compile_model
        self._logging = _logging
        self._signature = VQVAEGAN_Signature

        # self.optimizer_gen, self.optimizer_disc = self.configure_optimizers()
        # self.scheduler_gen, self.scheduler_disc = self.configure_schedulers()

    @property
    def signature(self):
        return self._signature

    def forward(self,):
        pass

    def training_step(self, batch: torch.Tensor, batch_idx) -> VQVAEGAN_Signature:
        images = batch["image"]

        # Get optimizers
        # opt_g, opt_d = self.optimizer_gen, self.optimizer_disc
        opt_g, opt_d = self.optimizers()
        
        # Get gradient scalers for mixed precision
        scaler = self.trainer.precision_plugin.scaler if hasattr(self.trainer.precision_plugin, "scaler") else None

        # Gradient accumulation
        is_accumulate_grad_batches = (batch_idx + 1) % self.manual_accumulate_grad_batches != 0

        # Generator training
        # Only zero gradients when we're about to step the optimizer
        if not is_accumulate_grad_batches:
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

        # Scale loss by accumulation steps for proper averaging
        loss_g = loss_g / self.manual_accumulate_grad_batches

        # Scale and backward
        if scaler:
            scaler.scale(loss_g).backward()
            # Only step when accumulation is complete
            if not is_accumulate_grad_batches:
                scaler.step(opt_g)
                scaler.update()
        else:
            self.manual_backward(loss_g)
            # Only step when accumulation is complete
            if not is_accumulate_grad_batches:
                opt_g.step()

        # Discriminator training
        discriminator_loss = torch.tensor(0.0, device=self.device)
        if self.current_epoch > self.autoencoder_warm_up_n_epochs:
            # Only zero gradients when we're about to step the optimizer
            if not is_accumulate_grad_batches:
                opt_d.zero_grad(set_to_none=True)

            # with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)

            logits_real = self.discriminator(images.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)

            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = self.adv_weight * discriminator_loss

            loss_d = loss_d / self.manual_accumulate_grad_batches

            self.manual_backward(loss_d)
            if not is_accumulate_grad_batches:
                if self.trainer.precision == 16:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                opt_d.step()
        
        if self._logging:
            self.log("train_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
            self.log("train_quantization_loss", quantization_loss, prog_bar=True, sync_dist=True)
            self.log("train_adv_loss", adversarial_loss, prog_bar=True, sync_dist=True)
            self.log("train_disc_loss", discriminator_loss, prog_bar=True, sync_dist=True)

        return {
            "recons_loss": recons_loss.detach(),
            "perceptual_loss": p_loss.detach(),
            "quantization_loss": quantization_loss.detach(),
            "adv_loss": adversarial_loss.detach(),
            "disc_loss": discriminator_loss.detach(),
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

            adversarial_loss = torch.tensor(1.0, device=self.device)
            if self.current_epoch > self.autoencoder_warm_up_n_epochs:
                # with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                adversarial_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)

                loss += self.adv_weight * adversarial_loss

        if self._logging:
            self.log("val_recons_loss", recons_loss, prog_bar=True, sync_dist=True)
            self.log("val_quantization_loss", quantization_loss, prog_bar=True, sync_dist=True)
            self.log("val_adv_loss", adversarial_loss, prog_bar=True, sync_dist=True)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return {
            "recons_loss": recons_loss.detach(),
            "perceptual_loss": p_loss.detach(),
            "quantization_loss": quantization_loss.detach(),
            "adv_loss": adversarial_loss.detach(),
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

        # scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt_gen, mode="min", factor=0.7, patience=10
        # )
        scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_gen,
            T_0=30,  # Restart every 30 epochs
            T_mult=2,  # Double the restart period each time
            eta_min=1e-7
        )
        scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_disc, mode="min", factor=0.7, patience=10
        )
        
        # return [opt_gen, opt_disc]
        return [
            {
                "optimizer": opt_gen,
                "lr_scheduler": {
                    "scheduler": scheduler_gen,
                    "interval": "epoch",
                    "frequency": 1,
                    # "monitor": "val_loss",
                }
            },
            {
                "optimizer": opt_disc,
                "lr_scheduler": {
                    "scheduler": scheduler_disc,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                }
            }
        ]
    
    # def configure_schedulers(self):
    #     """Configure schedulers for the optimizers."""

    #     # Get optimizers
    #     optimizer_gen, optimizer_disc = self.optimizers()

    #     scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer_gen, mode="min", factor=0.7, patience=10
    #     )
    #     scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer_disc, mode="min", factor=0.7, patience=10
    #     )
        
    #     return [scheduler_gen, scheduler_disc]

    def on_validation_epoch_end(self) -> None:
        """Handle scheduler stepping after validation."""
        # # Debug: Print available metrics
        # available_metrics = list(self.trainer.callback_metrics.keys())
        # console.log(f"[blue]Available metrics: {available_metrics}[/blue]")
        
        # # Print specific metric values
        # for metric in ["val_loss", "val_adv_loss", "val_recons_loss"]:
        #     value = self.trainer.callback_metrics.get(metric, None)
        #     if value is not None:
        #         console.log(f"[green]{metric}: {value:.4f}[/green]")
        #     else:
        #         console.log(f"[red]{metric}: Not found[/red]")

        optimizer_gen, optimizer_disc = self.optimizers()

        if not optimizer_gen or not optimizer_disc:
            console.log("[bold][red]Optimizers not found, cannot step schedulers.[/red][/bold]")
            return
        
        scheduler_gen, scheduler_disc = self.lr_schedulers()

        scheduler_gen.step()
        
        # Get validation loss and step schedulers
        val_loss = self.trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            # scheduler_gen.step(val_loss)
            scheduler_disc.step(val_loss)

            if self._logging:
                self.log("lr_g", self.optimizers()[0].param_groups[0]["lr"], prog_bar=True, sync_dist=True)
                self.log("lr_d", self.optimizers()[1].param_groups[0]["lr"], prog_bar=True, sync_dist=True)

            console.log(
                f"[bold][blue]Stepping schedulers: lr_g={scheduler_gen.optimizer.param_groups[0]['lr']:.6f}, "
                f"lr_d={scheduler_disc.optimizer.param_groups[0]['lr']:.6f}[/blue][/bold]"
            )

    
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