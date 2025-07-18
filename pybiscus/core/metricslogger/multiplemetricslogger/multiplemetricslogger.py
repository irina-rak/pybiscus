
from lightning.pytorch.loggers import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Union

class WandbMetricsLogger:
    def __init__(self, run):
        self.run = run
    def log_metrics(self, metrics, step=None):
        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

class NullMetricsLogger:

    def log_metrics(self, metrics, step=None):
        pass

class MultipleMetricsLogger(Logger):

    def __init__(self, metrics_loggers, save_dir: Union[str, Path] = "./logs"):
        super().__init__()
        # Wrap wandb runs with WandbMetricsLogger
        self.metrics_loggers = []
        for logger in metrics_loggers:
            if logger.__class__.__name__ == "Run" and hasattr(logger, "log"):
                self.metrics_loggers.append(WandbMetricsLogger(logger))
            else:
                self.metrics_loggers.append(logger)
        
        self._save_dir = Path(save_dir)
        self._version = 0

    @property
    def name(self) -> str:
        return "MultipleMetricsLogger"

    @property
    def version(self) -> Union[int, str]:
        return self._version

    @property
    def save_dir(self) -> str:
        return str(self._save_dir)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        # Log hyperparameters to all loggers that support it
        for metrics_logger in self.metrics_loggers:
            if hasattr(metrics_logger, 'log_hyperparams'):
                metrics_logger.log_hyperparams(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for metrics_logger in self.metrics_loggers:
            metrics_logger.log_metrics(metrics, step)

    def finalize(self, status: str) -> None:
        # Finalize all loggers that support it
        for metrics_logger in self.metrics_loggers:
            if hasattr(metrics_logger, 'finalize'):
                metrics_logger.finalize(status)
