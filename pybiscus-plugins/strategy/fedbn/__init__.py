
from typing import Dict, List, Tuple
from pydantic import BaseModel

from .fedbnstrategy import ConfigFabricFedBNStrategy, FabricFedBNStrategyFactory
from pybiscus.interfaces.flower.fabricstrategyfactory import FabricStrategyFactory

def get_modules_and_configs() -> Tuple[Dict[str, FabricStrategyFactory], List[BaseModel]]:

    registry = { "fedbn": FabricFedBNStrategyFactory, }
    configs  = [ConfigFabricFedBNStrategy,]

    return registry, configs
