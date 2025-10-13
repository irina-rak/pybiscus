from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar, Union

import torch.multiprocessing
import typer
import pybiscus.core.pybiscus_logger as logm
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from pybiscus.commands.apps_common import load_config
from pybiscus.core.metricslogger.file.filemetricslogger import FileMetricsLoggerFactory
from pybiscus.core.metricslogger.multiplemetricslogger.multiplemetricsloggerfactory import MultipleMetricsLoggerFactory
from pybiscus.flower_config.config_hardware import ConfigHardware
from pybiscus.plugin.registries import datamodule_registry, metricslogger_registry, model_registry, MetricsLoggerConfig, ModelConfig, DataConfig

torch.multiprocessing.set_sharing_strategy("file_system")

app = typer.Typer(pretty_exceptions_show_locals=False, rich_markup_mode="rich")

OmegaConf.register_new_resolver(
    "now",
    lambda pattern: datetime.now().strftime(pattern),
    replace=True
)


class ConfigCheckpointing(BaseModel):
    """A Pydantic Model to validate the Checkpointing configuration given by the user.

    Attributes
    ----------
    save_top_k: int = the number of best models to save
    monitor: str = the metric to monitor for saving the best model
    mode: str = the mode for monitoring (min or max)
    filename: str = the filename for saving the checkpoint
    every_n_epochs: int = save checkpoint every n epochs
    """

    save_top_k: int = 1
    monitor: Union[str, None] = None
    mode: str = "min"
    filename: str = "{epoch:02d}-{val_loss:.2f}"
    every_n_epochs: int = 1
    save_on_train_epoch_end: bool = True

class CustomModelCheckpoint(Callback):
    def __init__(
        self, 
        dirpath, 
        monitor, 
        mode="min",
        save_last=True, 
        save_top_k=-1, 
        every_n_epochs=1, 
        filename="epoch-{epoch:03d}-{val_loss:.3f}",
        save_on_train_epoch_end=True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.filename = filename
        self.save_on_train_epoch_end = save_on_train_epoch_end

        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.best_score = float("inf") if self.mode == "min" else -float("inf")
        self.topk_checkpoints = []

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if current is None:
            pl_module.print(f"Metric {self.monitor} not available; skipping checkpoint.")
            return

        # Compute filename and path
        filename = self.filename.format(epoch=epoch, **{self.monitor: current})
        save_path = self.dirpath / f"{filename}.ckpt"
        self.dirpath.mkdir(parents=True, exist_ok=True)

        # Always save 'last' if specified
        if self.save_last:
            last_path = self.dirpath / f"last_cp.ckpt"
            trainer.save_checkpoint(last_path)
        
        # Save best or all checkpoints as requested
        save_ckpt = False
        if self.save_top_k == -1:  # Save all
            save_ckpt = True
        else:
            is_better = (current < self.best_score) if (self.mode == "min") else (current > self.best_score)
            if is_better:
                self.best_score = current
                save_ckpt = True
                self.topk_checkpoints.append((current, save_path))
                # If limiting to top_k, enforce length
                if self.save_top_k > 0 and len(self.topk_checkpoints) > self.save_top_k:
                    # Remove the worst checkpoint
                    self.topk_checkpoints.sort(reverse=(self.mode == "min"))
                    _, worst_path = self.topk_checkpoints.pop()
                    if worst_path.exists():
                        worst_path.unlink()
        # Save checkpoint file if needed
        if save_ckpt:
            trainer.save_checkpoint(save_path)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # Optionally save on every train epoch end
        if self.save_on_train_epoch_end and ((trainer.current_epoch + 1) % self.every_n_epochs == 0):
            self.on_validation_end(trainer, pl_module)

class ConfigTrainerComputeContext(BaseModel):

    PYBISCUS_CONFIG: ClassVar[str] = "trainer"

    max_epochs: int

    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50

    hardware: ConfigHardware
    
    metrics_loggers: list[MetricsLoggerConfig()] # pyright: ignore[reportInvalidTypeForm]
    reporting_path: Path
    checkpointing: ConfigCheckpointing

    model_config = ConfigDict(extra="forbid")

class ConfigTrainer(BaseModel):
    """A Pydantic Model to validate the Trainer configuration given by the user.

    Attributes
    ----------
    root_dir:               str = the root directory for the experiment
    trainer:                ConfigTrainerComputeContext = the compute context for the trainer
    data:                   DataConfig = the data configuration
    model:                  ModelConfig = the model configuration
    """

    PYBISCUS_ALIAS: ClassVar[str] = "Pybiscus trainer configuration"

    root_dir:               str
    trainer:                ConfigTrainerComputeContext
    data:                   DataConfig() # pyright: ignore[reportInvalidTypeForm]
    model:                  ModelConfig() # pyright: ignore[reportInvalidTypeForm]

    model_config = ConfigDict(extra="forbid")

def check_and_build_trainer_config(conf_loaded: dict) -> ConfigTrainer :

    logm.console.log(conf_loaded)
    _conf = ConfigTrainer(**conf_loaded)
    logm.console.log(_conf)
        
    return _conf

@app.callback()
def local():
    """The local part of Pybiscus.

    Train locally the model.
    """


@app.command(name="launch")
def launch_config(config: Annotated[Path, typer.Argument()] = None):
    """Launch a local training.

    This function is here mostly for prototyping and testing models on local data, without the burden of potential Federated Learning issues.
    It is simply a re implementation of the Lightning CLI, adapted for Pybiscus.

    Parameters
    ----------
    config : Path
        the path to the configuration file.

    Raises
    ------
    typer.Abort
        _description_
    typer.Abort
        _description_
    typer.Abort
        _description_
    """

    # handling mandatory config path parameter

    conf_loaded = load_config(config)
    conf_loaded = OmegaConf.to_container(conf_loaded, resolve=True)

    conf = check_and_build_trainer_config(conf_loaded)

    # Metrics loggers
    _metricslogger_classes = [ metricslogger_registry()[mlogger.name](config=mlogger.config) for mlogger in conf.trainer.metrics_loggers ]
    _file_metrics_logger_factory = FileMetricsLoggerFactory( "metrics.txt" ) # additional factory for logging metrics into reporting directory
    _metricslogger_classes.append( _file_metrics_logger_factory ) # add this factory to the list
    _metricslogger = MultipleMetricsLoggerFactory(_metricslogger_classes).get_metricslogger(conf.trainer.reporting_path)

    # Model
    model_class = model_registry()[conf.model.name]
    model = model_class(**conf.model.config.model_dump())

    # Load the model weights if provided
    if getattr(conf.model, 'pretrained_weights_path', None) is not None:
        weights_path = Path(conf.model.pretrained_weights_path)
        if not weights_path.exists():
            logm.console.print(f"[red]Weights path {weights_path} does not exist.[/red]")
            raise typer.Abort()
        logm.console.print(f"[blue]Loading model weights from {weights_path}...[/blue]")
        try:
            # State dict is in 'state_dict' key for Lightning models
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu")["state_dict"],
                strict=False  # allow or not missing keys in the state dict
            )
        except Exception as e:
            logm.console.print(f"[red]Error loading model weights: {e}[/red]")
            raise typer.Abort()

    # Data
    # data = datamodule_registry()[conf_loaded["data"]["name"]](**conf_loaded["data"]["config"])
    data_class = datamodule_registry()[conf.data.name]
    data = data_class(**conf.data.config.model_dump())

    checkpoint_dir = Path(conf.trainer.reporting_path) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        RichModelSummary(),
        RichProgressBar(theme=RichProgressBarTheme(metrics="blue")),
        # CustomModelCheckpoint(
        #     dirpath=checkpoint_dir,
        #     save_last=True,
        #     **conf.trainer.checkpointing.model_dump()  # type: ignore[call-arg] -- this is a dict, not a class
        # )
    ]

    trainer = Trainer(
        # default_root_dir=Path(conf.root_dir) / f"experiments/local/",
        default_root_dir=Path(conf.trainer.reporting_path),
        enable_checkpointing=True,
        logger=_metricslogger,
        max_epochs=conf.trainer.max_epochs,
        accumulate_grad_batches=conf.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=conf.trainer.check_val_every_n_epoch,
        log_every_n_steps=conf.trainer.log_every_n_steps,
        callbacks=callbacks,
        **conf.trainer.hardware.model_dump()  # type: ignore[call-arg] -- this is a dict, not a class
    )

    trainer.fit(model, data)

    if conf.trainer.checkpointing.save_on_train_epoch_end:
        # Save the final model configuration to a YAML file
        logm.console.print(f"[blue]Saving final model configuration to {trainer.log_dir}/config_launch.yml...[/blue]")
        trainer.save_checkpoint(trainer.default_root_dir + f"/{conf.trainer.metrics_loggers[0].config.params.name}.ckpt")

    with open(trainer.log_dir + "/config_launch.yml", "w") as file:
        OmegaConf.save(config=conf_loaded, f=file)


if __name__ == "__main__":
    app()
