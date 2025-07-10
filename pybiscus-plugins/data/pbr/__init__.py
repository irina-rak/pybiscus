
from typing import Dict, List, Tuple
import lightning.pytorch as pl
from pydantic import BaseModel

from pbr.pbr_datamodule import ConfigData_PBR, PBRLitDataModule

def get_modules_and_configs() -> Tuple[Dict[str, pl.LightningDataModule], List[BaseModel]]:

    registry = {"pbr": PBRLitDataModule}
    configs  = [ConfigData_PBR]

    return registry, configs
