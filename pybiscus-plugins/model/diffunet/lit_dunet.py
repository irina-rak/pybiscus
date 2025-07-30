from pathlib import Path
from typing import List, Literal, Sequence, TypedDict, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from monai.inferers import LatentDiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism
from pydantic import BaseModel, ConfigDict
from torch.amp import autocast

from pybiscus.core.pybiscus_logger import console
from klae.lit_klae import ConfigKLAutoEncoder, ConfigAlexDiscriminator, LitKLAutoEncoder
from vqvae.lit_vqvae import ConfigVQAutoEncoder, LitVQVAE


class ConfigDiffusionUnet(BaseModel):
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
    in_channels: int # This should match autoencoder's embedding_dim
    out_channels: int # This should match autoencoder's embedding_dim
    num_res_blocks: int
    channels: Sequence[int]
    attention_levels: Sequence[int]
    norm_num_groups: int
    num_head_channels: Union[int, Sequence[int]]

    model_config = ConfigDict(extra="forbid")


class ConfigScheduler(BaseModel):
    """A Pydantic Model to validate the scheduler config given by the user.

    Attributes
    ----------
    num_train_timesteps: int
        number of training timesteps
    beta_start: float
        starting beta value for the scheduler
    beta_end: float
        ending beta value for the scheduler
    beta_schedule: Literal["linear", "cosine"]
        type of beta schedule to use
    """

    num_train_timesteps: int = 1000
    schedule: Literal["linear_beta", "scaled_linear_beta", "cosine"] = "linear_beta"
    beta_start: float = 0.0015
    beta_end: float = 0.0195

    model_config = ConfigDict(extra="forbid")


class ConfigUnet(BaseModel):
    """A Pydantic Model to validate the LitPBR config given by the user.

    Attributes
    ----------
    unet_config: ConfigDiffusionUnet
        configuration for the UNet model
    scheduler_config: ConfigScheduler
        configuration for the scheduler
    lr: float
        learning rate for the optimizer
    autoencoder_warm_up_n_epochs: int
        number of epochs to warm up the autoencoder
    seed: Union[int, None]
        random seed for reproducibility
    _logging: bool
        whether to log training information
    """

    unet_config: ConfigDiffusionUnet
    scheduler_config: ConfigScheduler
    generator_config: ConfigVQAutoEncoder
    lr: float = 1e-4
    autoencoder_weights: Union[str, Path]
    seed: Union[int, None] = None
    _logging: bool = True

    model_config = ConfigDict(extra="forbid")


class ConfigModel_DiffusionUnet(BaseModel):
    """Pydantic BaseModel to validate Configuration for DiffusionUnet model.

    Attributes
    ----------
    name: Literal["dunet"]
        designation "dunet" to choose
    config:
        configuration for the model LitUnet
    """

    name: Literal["dunet"]
    pretrained_weights_path: Union[str, None] = None
    config: ConfigUnet

    model_config = ConfigDict(extra="forbid")


class DUSignature(TypedDict):
    """A TypedDict to represent the signature of both training and validation steps of LDM model.

    Used in particular in train and test loops, to gather information on how many metrics are returned by the model.

    Attributes
    ----------
    loss: torch.Tensor
        The main loss
    """

    loss: torch.Tensor


class LitDiffusionUnet(pl.LightningModule):
    def __init__(
        self,
        unet_config: ConfigDiffusionUnet,
        scheduler_config: ConfigScheduler,
        generator_config: Union[ConfigKLAutoEncoder, ConfigVQAutoEncoder],
        lr: float = 1e-4,
        autoencoder_weights: Union[str, Path] = None,
        seed: Union[int, None] = None,
        _logging: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        if seed is not None and type(seed) is int:
            set_determinism(seed=seed)
            console.print(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

        self.autoencoder = LitVQVAE(
            vqvae_config=ConfigVQAutoEncoder(**generator_config),
        )
        try:
            self.autoencoder.load_state_dict(torch.load(autoencoder_weights, map_location=self.device)["state_dict"])
        except Exception as e:
            console.log(f"[bold][red]Error loading autoencoder weights: {e}[/red][/bold]")
            raise e
        self.autoencoder = self.autoencoder.vqvae
        self.autoencoder.eval()  # That way, there's no gradient running around when the diffusion model is trained

        # Explicitly freeze autoencoder parameters to avoid DDP unused parameter warnings
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.model = DiffusionModelUNet(
            **(unet_config.model_dump() if hasattr(unet_config, "model_dump") else unet_config)
        )

        self.scaling_factor = 1.0

        self.scheduler = DDPMScheduler(
            **(scheduler_config.model_dump() if hasattr(scheduler_config, "model_dump") else scheduler_config)
        )

        self.inferer = LatentDiffusionInferer(
            scheduler=self.scheduler,
        )

        self.lr = lr
        self._logging = _logging
        self._signature = DUSignature

    @property
    def signature(self):
        return self._signature

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def training_step(self, batch: torch.Tensor, batch_idx) -> DUSignature:
        images = batch["image"]
        
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
            z = self.autoencoder.encode_stage_2_inputs(
                images.float()
            )
            # Scale the latent representation
            z = z * self.scaling_factor

        if self.current_epoch == 0 and self.global_step == 0:
            # Better scaling factor calculation
            self.scaling_factor = 1 / (torch.std(z) + 1e-8)

            # Only clamp if values are genuinely problematic (very extreme)
            if self.scaling_factor > 10.0 or self.scaling_factor < 0.01:
                console.print(f"[bold][yellow]Warning: Extreme scaling factor {self.scaling_factor:.4f}, using fallback[/yellow][/bold]")
                self.scaling_factor = torch.clamp(self.scaling_factor, min=0.01, max=10.0)

            self.inferer.scaling_factor = self.scaling_factor
            console.print(f"[bold][blue]Scaling factor set to: {self.scaling_factor}[/blue][/bold]")
            console.print(f"[bold][blue]Latent representation shape: {z.shape}[/blue][/bold]")
            console.print(f"[bold][blue]Latent mean: {torch.mean(z):.4f}, std: {torch.std(z):.4f}[/blue][/bold]")

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
            # Check for NaN in latent representation
            if torch.isnan(z).any():
                console.print("[bold][red]NaN detected in latent representation![/red][/bold]")
                return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

            noise = torch.randn_like(z)
            
            # Use more diverse timestep sampling
            timesteps = torch.randint(
                0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Add noise to latent representation
            noisy_latents = self.inferer.scheduler.add_noise(z, noise, timesteps)

            # Predict the noise
            noise_pred = self.model(
                noisy_latents,
                timesteps
            )

            # Check for NaN in predictions
            if torch.isnan(noise_pred).any():
                console.print("[bold][red]NaN detected in noise prediction![/red][/bold]")
                return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

            if self.global_step < 1000:  # Warm-up phase
                loss = F.mse_loss(noise_pred.float(), noise.float())
            else:
                # Huber loss for better stability
                loss = F.smooth_l1_loss(noise_pred.float(), noise.float())

        if self._logging:
            self.log("train_loss", loss, prog_bar=True)
            self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, sync_dist=True)
            
            # Log additional metrics
            self.log("noise_pred_mean", torch.mean(noise_pred), prog_bar=False, sync_dist=True)
            self.log("noise_pred_std", torch.std(noise_pred), prog_bar=False, sync_dist=True)
            self.log("target_noise_mean", torch.mean(noise), prog_bar=False, sync_dist=True)
            self.log("target_noise_std", torch.std(noise), prog_bar=False, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx) -> DUSignature:
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
            with torch.no_grad():
                images = batch["image"]

                z = self.autoencoder.encode_stage_2_inputs(
                    images.float()
                )

                noise = torch.randn_like(z)
                timesteps = torch.randint(
                    0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                noise_pred = self.inferer(
                    inputs=images,
                    autoencoder_model=self.autoencoder,
                    diffusion_model=self.model,
                    noise=noise,
                    timesteps=timesteps
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

        if self._logging:
            self.log("val_loss", loss, prog_bar=True)

        return {"loss": loss}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> DUSignature:
        # with torch.no_grad():
        #     images = batch["image"]
            
        #     reconstruction, _, _ = self.autoencoder(images)
        #     recons_loss = F.l1_loss(reconstruction.float(), images.float())
                
        # if self._logging:
        #     self.log("val_recons_loss", recons_loss, prog_bar=True)
            
        # return {
        #     "recons_loss": recons_loss.detach(),
        # }
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizers with learning rate scheduling.

        Returns
        -------
        dict
            Dictionary containing optimizer and scheduler configuration.
        """
        # Try AdamW with better hyperparameters for diffusion models
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-6,
            eps=1e-8
        )
        
        # Use cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Restart every 50 epochs
            T_mult=2,  # Double the restart period each time
            eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
            # Add gradient clipping - crucial for diffusion models
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
        }
    
    # def on_train_epoch_end(self) -> None:
    #     """Hook called at the end of each training epoch."""
    #     scheduler = self.lr_schedulers()

    #     if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #         # Update the scheduler with the latest validation loss
    #         val_loss = self.trainer.callback_metrics.get("val_loss", None)
    #         if val_loss is not None:
    #             scheduler.step(val_loss.item())

    #     elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    #         # Step the scheduler at the end of each epoch
    #         scheduler.step()

    #     self.log("learning_rate", self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)