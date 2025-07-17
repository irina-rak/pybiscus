# Logging: Tensorboard and WandB

## Tensorboard

The Pybiscus implementation of the FabricStrategy allows for simple logging of the losses and metrics of all Clients, and the loss and metrics of the evaluation of the Server on the (global) test dataset, if provided. The Fabric instance on the Server side can use a Tensorboard logger, and everything is then simply handled.

The Tensorboard logs are by default in `conf["root_dir"] + conf["logger"]["subdir"]`. Fabric handles the versioning of the FL session. You can view the Tensorboards by launching a tensorboard session:
```bash
(.venv) tensorboard --logdir path-to-experiments --bind_all --port your-port
```
where `path-to-experiments` is `conf["root_dir"]` and `your-port` is the port of your choice for the Tensorboard server.

## WandB (Weights & Biases)

Pybiscus also supports logging to [Weights & Biases (wandb)](https://wandb.ai/) from the server side. You can enable this by specifying `wandb` as the metrics logger in your server configuration YAML (see example below). Metrics and losses will be logged to your wandb project for easy experiment tracking and visualization.

To use wandb logging, set up your config like this:
```yaml
server_compute_context:
  metrics_loggers:
    - name: wandb
      config:
        api_key_definition:
          env_var_name: WANDB_API_KEY
        params:
          project_name: "Pybiscus-FL"
          entity_name: "your-entity"
          run_name: "server-run"
          run_group: "fl-group"
          job_type: "server"
          is_client: false
```
Make sure your `WANDB_API_KEY` is set in your environment.

## Logging

Pybiscus also uses Rich and its Console to log info during the FL session, visible in the terminal. Typer uses Rich too, especially to print errors nicely. Flower logs information as well, dedicated in particular to the gRPC communications and the process between Server and Clients.
