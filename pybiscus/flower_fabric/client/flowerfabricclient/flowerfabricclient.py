from collections import OrderedDict
from collections.abc import Mapping
import flwr as fl
from lightning import Fabric, LightningDataModule, LightningModule
import torch
import torch.nn as nn
import numpy as np

from pybiscus.flower_config.config_computecontext import ConfigClientComputeContext
from pybiscus.ml.loops_fabric import test_loop, train_loop
import pybiscus.core.pybiscus_logger as logm

def parse_optimizers(lightning_optimizers):
    """
    Parse the output of lightning configure_optimizers
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
    To extract only the optimizers (and not the lr_schedulers)
    """
    optimizers = []
    if lightning_optimizers:
        if isinstance(lightning_optimizers, Mapping):
            optimizers.append(lightning_optimizers['optimizer']) 
        elif isinstance(lightning_optimizers, torch.optim.Optimizer):
            optimizers.append(lightning_optimizers)
        else:
            for optmizers_conf in lightning_optimizers:
                if isinstance(optmizers_conf, dict):
                    optimizers.append(lightning_optimizers)
                else:
                    optimizers.append(optmizers_conf)
    return optimizers

class FlowerFabricClient(fl.client.NumPyClient):
    """A Fabric-based, modular Flower Client.

    The present FlowerClient override the usual Flower Client, by using Fabric as a backbone.
    The Client now holds data, the model and a Fabric instance which takes care of everything regarding
    hardware, precision and such.

    """

    def __init__(
        self,
        cid: int,
        model: LightningModule,
        data: LightningDataModule,
        num_examples: dict[str, int],
        conf_fabric: ConfigClientComputeContext,
        pre_train_val: bool = False,
        agg_bn: bool = True,
    ) -> None:
        """Initialize the FlowerClient instance.

        Override the usual fl.client.NumPyClient configuration and add data, model and fabric attributes.

        Parameters
        ----------
        cid : int
            the client identifier
        model : LightningModule
            the model used for the FL training
        data : LightningDataModule
            the data used for the training/validation process
        num_examples : dict[str, int]
            needed by Flower, for the FedAvg Streategy typically
        conf_fabric : ConfigClientComputeContext
            a Pydantic-validated configuration for the Fabric instance
        """
        super().__init__()
        self.cid = cid
        self.model = model
        self.data = data

        self.conf_fabric = conf_fabric.model_dump()
        self.num_examples = num_examples
        self.pre_train_val = pre_train_val
        self.agg_bn = agg_bn

        if not self.agg_bn:
            logm.console.log(
                f"[blue]Client {self.cid} will NOT aggregate BatchNorm and Bias parameters[/blue]"
            )

        self.optimizers = parse_optimizers(self.model.configure_optimizers())

        self.fabric = Fabric(**self.conf_fabric)

    def initialize(self):
        self.fabric.launch()

        if hasattr(self, "optimizers") and self.optimizers:
            self.model, *self.optimizers = self.fabric.setup(self.model, *self.optimizers)
        else:
            self.model = self.fabric.setup(self.model)

        (
            self._train_dataloader,
            self._validation_dataloader,
        ) = self.fabric.setup_dataloaders(
            self.data.train_dataloader(), self.data.val_dataloader()
        )

    # def get_parameters(self, config):
    #     logm.console.log(f"[Client] get_parameters, config: {config}")
    #     if self.agg_bn:
    #         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    #     else:
    #         return [
    #             val.cpu().numpy()
    #             for name, val in self.model.state_dict().items()
    #             if "bn" not in name
    #         ]

    def get_parameters(self, config):
        logm.console.log(f"[Client] get_parameters, config: {config}")
        if self.agg_bn:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        # Get all parameter names from the model
        all_param_names = list(self.model.state_dict().keys())
        
        # Find batch norm layer parameter names
        bn_param_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                # Add all parameters belonging to this batch norm layer
                for param_name in all_param_names:
                    if param_name.startswith(name + '.'):
                        bn_param_names.add(param_name)
        
        # Filter out batch norm parameters and return as numpy arrays
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if name not in bn_param_names
        ]

    # def set_parameters(self, parameters):
    #     logm.console.log("[Client] set_parameters")
    #     if self.agg_bn:
    #         params_dict = zip(self.model.state_dict().keys(), parameters)
    #     else:
    #         params_dict = zip(
    #             (k for k in self.model.state_dict().keys() if "bn" not in k),
    #             parameters,
    #         )
    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #     self.model.load_state_dict(state_dict, strict=True)

    def set_parameters(self, parameters):
        if self.agg_bn:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            return
        
        # Get all parameter names from the model
        all_param_names = list(self.model.state_dict().keys())
        
        # Find batch norm layer parameter names
        bn_param_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                # Add all parameters belonging to this batch norm layer
                for param_name in all_param_names:
                    if param_name.startswith(name + '.'):
                        bn_param_names.add(param_name)
        
        # Filter out batch norm parameters
        filtered_param_names = [name for name in all_param_names if name not in bn_param_names]
        
        # Ensure we have the right number of parameters
        if len(filtered_param_names) != len(parameters):
            print(f"Warning: Expected {len(filtered_param_names)} parameters, got {len(parameters)}")
            print(f"Excluded {len(bn_param_names)} batch norm parameters: {sorted(bn_param_names)}")
        
        # Create state dict with filtered parameters
        params_dict = zip(filtered_param_names, parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow missing BN params

    def fit(self, parameters, config):
        logm.console.log(f"[Client {self.cid}] fit, config: {config}")
        self.set_parameters(parameters)
        metrics = {}

        if self.pre_train_val:
            logm.console.log(
                f"Round {config['server_round']}, pre train validation started..."
            )
            results_pre_train = test_loop(
                self.fabric, self.model, self._validation_dataloader
            )
            for key, val in results_pre_train.items():
                metrics[f"{key}_pre_train_val"] = val

        logm.console.log(f"Round {config['server_round']}, training Started...")

        results_train = train_loop(
            self.fabric,
            self.model,
            self._train_dataloader,
            self.optimizers, # Alice TODO extend this to multiple optimizers ??
            epochs=config["local_epochs"],
        )
            
        logm.console.log(f"Training Finished! Loss is {results_train['loss']}")
        metrics["cid"] = self.cid
        for key, val in results_train.items():
            metrics[key] = val
        return self.get_parameters(config={}), self.num_examples["trainset"], metrics

    def evaluate(self, parameters, config):
        logm.console.log(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        metrics = {}
        logm.console.log(f"Round {config['server_round']}, evaluation Started...")
        results_evaluate = test_loop(
            self.fabric, self.model, self._validation_dataloader
        )
        logm.console.log(
            f"Evaluation finished! Loss is {results_evaluate['loss']}, metric {list(results_evaluate.keys())[0]} is {results_evaluate[list(results_evaluate.keys())[0]]}"
        )
        metrics["cid"] = self.cid
        for key, val in results_evaluate.items():
            metrics[key] = val
        return results_evaluate["loss"], self.num_examples["valset"], metrics
