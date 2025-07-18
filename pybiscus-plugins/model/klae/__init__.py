
from typing import Dict, List, Tuple
import lightning.pytorch as pl
from pydantic import BaseModel

from klae.lit_ldm import ( LitKLAutoEncoder, ConfigModel_KLAE, )

def get_modules_and_configs() -> Tuple[Dict[str, pl.LightningModule], List[BaseModel]]:

    registry = { "klae":  LitKLAutoEncoder, }
    configs  = [ConfigModel_KLAE]

    return registry, configs
