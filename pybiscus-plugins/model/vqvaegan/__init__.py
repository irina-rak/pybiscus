
from typing import Dict, List, Tuple
import lightning.pytorch as pl
from pydantic import BaseModel

from vqvaegan.lit_vqvaegan import ( LitVQVAEGAN, ConfigModel_VQVAEGAN, )

def get_modules_and_configs() -> Tuple[Dict[str, pl.LightningModule], List[BaseModel]]:

    registry = { "vqvaegan":  LitVQVAEGAN }
    configs  = [ConfigModel_VQVAEGAN]

    return registry, configs
