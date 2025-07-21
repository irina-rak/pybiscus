from pathlib import Path
from typing import List, Literal, TypedDict, Union

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
    in_channels: int
    out_channels: int
    num_res_blocks: int
    channels: List[int]
    attention_levels: List[int]
    norm_num_groups: int
    num_head_channels: List[int]

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
    schedule: Literal["linear_beta", "cosine"] = "linear_beta"
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
    generator_config: ConfigKLAutoEncoder
    discriminator_config: ConfigAlexDiscriminator
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
        generator_config: ConfigKLAutoEncoder,
        discriminator_config: ConfigAlexDiscriminator,
        lr: float = 1e-4,
        autoencoder_weights: Union[str, Path] = None,
        seed: Union[int, None] = None,
        _logging: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        if seed is not None and type(seed) is int:
            set_determinism(seed=seed)
            console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

        # Memory optimization settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()

        # Disable automatic optimization for manual control
        self.automatic_optimization = False

        self.autoencoderkl = LitKLAutoEncoder(
            # **(generator_config.model_dump() if hasattr(generator_config, "model_dump") else generator_config)
            generator_config=generator_config,
            discriminator_config=discriminator_config
        )
        try:
            self.autoencoderkl.load_state_dict(torch.load(autoencoder_weights, map_location=self.device)["state_dict"])
        except Exception as e:
            console.log(f"[bold][red]Error loading autoencoder weights: {e}[/red][/bold]")
            raise e
        self.autoencoder = self.autoencoderkl.autoencoderkl
        self.autoencoder.eval()  # That way, there's no gradient running around when the diffusion model is trained

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

        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        
        if self.current_epoch == 0 and self.global_step == 0:
            # console.log("[bold][blue]Scaling factor calculation...[/blue][/bold]")
            with torch.no_grad():
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
                    self.z = self.autoencoder.encode_stage_2_inputs(
                        images.float()
                    )
                    self.scaling_factor = 1 / torch.std(self.z)
                    self.inferer.scaling_factor = self.scaling_factor
            console.log(f"[bold][blue]Scaling factor set to: {self.scaling_factor}[/blue][/bold]")

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True):
            noise = torch.randn_like(self.z)

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

        self.manual_backward(loss)
        optimizer.step()

        if self._logging:
            self.log("train_loss", loss, prog_bar=True)

        return {"loss": loss.detach()}

    def validation_step(self, batch: torch.Tensor, batch_idx) -> DUSignature:
        pass

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> DUSignature:
        # with torch.no_grad():
        #     images = batch["image"]
            
        #     reconstruction, _, _ = self.autoencoderkl(images)
        #     recons_loss = F.l1_loss(reconstruction.float(), images.float())
                
        # if self._logging:
        #     self.log("val_recons_loss", recons_loss, prog_bar=True)
            
        # return {
        #     "recons_loss": recons_loss.detach(),
        # }
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizers for the generator and discriminator.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for the model.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)