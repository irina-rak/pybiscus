
from typing import Dict, List, Tuple
import lightning.pytorch as pl
from pydantic import BaseModel

from diffunet.lit_dunet import ( LitDiffusionUnet, ConfigModel_DiffusionUnet, )

def get_modules_and_configs() -> Tuple[Dict[str, pl.LightningModule], List[BaseModel]]:

    registry = { "dunet": LitDiffusionUnet }
    configs  = [ConfigModel_DiffusionUnet]

    return registry, configs
