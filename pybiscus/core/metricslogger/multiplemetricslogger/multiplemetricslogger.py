
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

class MultipleMetricsLogger( ):

    def __init__(self, metrics_loggers):
        # Wrap wandb runs with WandbMetricsLogger
        self.metrics_loggers = []
        for logger in metrics_loggers:
            if logger.__class__.__name__ == "Run" and hasattr(logger, "log"):
                self.metrics_loggers.append(WandbMetricsLogger(logger))
            else:
                self.metrics_loggers.append(logger)

    def log_metrics(self, metrics, step=None):
        for metrics_logger in self.metrics_loggers:
            metrics_logger.log_metrics(metrics, step)
