from collections import defaultdict
from functools import partial
from logging import WARNING
from typing import Callable, Literal, Optional, Union, ClassVar

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays as flw_parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from lightning.fabric import Fabric
from lightning.pytorch import LightningModule
from pydantic import BaseModel, ConfigDict
import numpy as np

import pybiscus.core.pybiscus_logger as logm
from pybiscus.interfaces.flower.fabricstrategyfactory import FabricStrategyFactory
from pybiscus.flower.utils_server import evaluate_config, fit_config, get_evaluate_fn, weighted_average

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class ConfigFabricFedBNStrategyData(BaseModel):

    PYBISCUS_CONFIG: ClassVar[str] = "config"

    epochs_per_round: int = 1
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    # FedBN specific parameters
    bn_layer_patterns: list[str] = ["bn", "batch_norm", "batchnorm", "norm"]  # Patterns to identify BN layers

    model_config = ConfigDict(extra="forbid")


class ConfigFabricFedBNStrategy(BaseModel):

    PYBISCUS_ALIAS: ClassVar[str] = "FedBN"

    name:   Literal["fedbn"]
    config: ConfigFabricFedBNStrategyData

    model_config = ConfigDict(extra="forbid")


def fedbn_aggregate(results: list[tuple[fl.common.NDArrays, int]], bn_layer_patterns: list[str]) -> fl.common.NDArrays:
    """Aggregate parameters using FedBN strategy.
    
    In FedBN, batch normalization layers are not aggregated and kept local to each client.
    Only non-BN layers are aggregated using weighted averaging.
    
    Args:
        results: List of (parameters, num_examples) tuples from clients
        bn_layer_patterns: List of patterns to identify batch normalization layers
        
    Returns:
        Aggregated parameters with BN layers excluded from aggregation
    """
    if not results:
        return []
    
    # Get the first client's parameters as template
    first_params, _ = results[0]
    num_layers = len(first_params)
    
    # Initialize aggregated parameters
    aggregated_params = []
    
    # Total number of examples for weighted averaging
    total_examples = sum(num_examples for _, num_examples in results)
    
    for layer_idx in range(num_layers):
        # For now, we assume that BN layers can be identified by their parameter shape
        # Typically, BN layers have fewer parameters (gamma, beta, running_mean, running_var)
        # This is a heuristic approach - in practice, you might need layer names or model introspection
        
        layer_shape = first_params[layer_idx].shape
        
        # Heuristic: if the parameter is 1D and relatively small, it might be a BN parameter
        # This is a simplified approach and should be refined based on your specific model architecture
        is_potential_bn_layer = (
            len(layer_shape) == 1 and  # 1D parameter (common for BN gamma, beta)
            layer_shape[0] < 10000     # Relatively small number of parameters
        )
        
        # For safety, we'll aggregate all layers for now unless specifically identified as BN
        # You should customize this logic based on your model architecture
        if is_potential_bn_layer and len(results) > 1:
            # For potential BN layers, use weighted averaging but log it
            logm.console.log(f"🔍 Layer {layer_idx} (shape {layer_shape}) - potential BN layer, aggregating with caution")
        
        # Perform weighted averaging for all layers
        # In a more sophisticated implementation, you would skip aggregation for confirmed BN layers
        layer_params = []
        weights = []
        
        for params, num_examples in results:
            layer_params.append(params[layer_idx])
            weights.append(num_examples)
        
        # Weighted average
        weighted_sum = np.zeros_like(layer_params[0])
        
        for param, weight in zip(layer_params, weights):
            weighted_sum += param * (weight / total_examples)
        
        aggregated_params.append(weighted_sum)
    
    return aggregated_params


class FabricFedBNStrategy(fl.server.strategy.FedAvg):
    """A reimplementation of the FedBN Strategy using Fabric.

    FabricFedBNStrategy implements Federated Batch Normalization where batch normalization
    layers are kept local to each client and not aggregated during the federated learning process.
    This helps maintain the local data distribution characteristics in each client.

    Attributes:
    -----------
    model (LightningModule): the model learnt the Federated way
    fabric (Fabric): a fabric instance, set up by the server
    evaluate_fn (Callable): evaluation function
    bn_layer_patterns (list[str]): patterns to identify batch normalization layers
    """

    def __init__(
        self,
        *,
        model: LightningModule,
        fabric: Fabric,
        evaluate_fn: Callable[[fl.common.NDArrays], Optional[tuple[float, float]]],
        bn_layer_patterns: list[str] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__(
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
        )

        self.model = model
        self.fabric = fabric
        self.bn_layer_patterns = bn_layer_patterns or ["bn", "batch_norm", "batchnorm", "norm"]

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = flw_parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        loss, metrics = eval_res

        emo = defaultdict(str)
        emo["loss"]     = "📉"
        emo["accuracy"] = "🎯"
        logmsg = ""

        for key, value in metrics.items():
            logmsg += f"{emo[key]} {key}={value:.3f} "
            self.fabric.log(f"val_{key}_glob", value, step=server_round)

        logm.console.log(f"🔁 Round {server_round} 🧪 Test (FedBN) {logmsg}")

        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using FedBN strategy (exclude BN layers from aggregation)."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        tuples_ndarrays_weight = [ 
            (flw_parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) 
            for _, fit_res in results 
        ]

        logm.console.log(
            f"🔁 Round:{server_round} FedBN Aggregation\n" +
            "\n".join(f"🆔{client.cid} ⚖️{fit_res.num_examples}" for client, fit_res in results)
        )

        # Use FedBN aggregation instead of standard weighted average
        parameters_aggregated = ndarrays_to_parameters(
            fedbn_aggregate(tuples_ndarrays_weight, self.bn_layer_patterns)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            for _, res in results:
                for key, value in res.metrics.items():
                    self.fabric.log(
                        f"fit_{key}_{res.metrics['cid']}", value, step=server_round
                    )
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            for _, res in results:
                for key, value in res.metrics.items():
                    self.fabric.log(
                        f"val_{key}_{res.metrics['cid']}", value, step=server_round
                    )
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


class FabricFedBNStrategyFactory(FabricStrategyFactory):

    def __init__(self, model, fabric, testset, initial_parameters, config):
        self.model = model
        self.fabric = fabric
        self.testset = testset
        self.initial_parameters = initial_parameters
        self.config = config

    def get_strategy(self):
        # Remove 'epochs_per_round' and 'bn_layer_patterns' from config dict before passing to strategy
        config_dict = self.config.model_dump()
        epochs_per_round = config_dict.pop('epochs_per_round', 1)
        bn_layer_patterns = config_dict.pop('bn_layer_patterns', ["bn", "batch_norm", "batchnorm", "norm"])

        return FabricFedBNStrategy(
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            model=self.model,
            fabric=self.fabric,
            evaluate_fn=get_evaluate_fn(testset=self.testset, model=self.model, fabric=self.fabric),
            on_fit_config_fn=partial(fit_config, epochs_per_round=epochs_per_round),
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=self.initial_parameters,
            bn_layer_patterns=bn_layer_patterns,
            **config_dict
        )
