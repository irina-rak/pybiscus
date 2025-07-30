from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar

import torch.multiprocessing
import typer
import pybiscus.core.pybiscus_logger as logm
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
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


class ConfigTrainerComputeContext(BaseModel):

    PYBISCUS_CONFIG: ClassVar[str] = "trainer"

    max_epochs: int

    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50

    hardware: ConfigHardware
    
    metrics_loggers: list[MetricsLoggerConfig()] # pyright: ignore[reportInvalidTypeForm]
    reporting_path: Path

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
                strict=True  # allow missing keys in the state dict
            )
        except Exception as e:
            logm.console.print(f"[red]Error loading model weights: {e}[/red]")
            raise typer.Abort()

    # Data
    # data = datamodule_registry()[conf_loaded["data"]["name"]](**conf_loaded["data"]["config"])
    data_class = datamodule_registry()[conf.data.name]
    data = data_class(**conf.data.config.model_dump())

    trainer = Trainer(
        # default_root_dir=Path(conf.root_dir) / f"experiments/local/",
        default_root_dir=Path(conf.trainer.reporting_path),
        enable_checkpointing=True,
        logger=_metricslogger,
        max_epochs=conf.trainer.max_epochs,
        accumulate_grad_batches=conf.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=conf.trainer.check_val_every_n_epoch,
        log_every_n_steps=conf.trainer.log_every_n_steps,
        callbacks=[
            RichModelSummary(),
            RichProgressBar(theme=RichProgressBarTheme(metrics="blue")),
        ],
        **conf.trainer.hardware.model_dump()  # type: ignore[call-arg] -- this is a dict, not a class
    )

    trainer.fit(model, data)
    with open(trainer.log_dir + "/config_launch.yml", "w") as file:
        OmegaConf.save(config=conf_loaded, f=file)


if __name__ == "__main__":
    app()
