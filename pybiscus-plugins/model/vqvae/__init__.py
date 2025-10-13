
from typing import Dict, List, Tuple
import lightning.pytorch as pl
from pydantic import BaseModel

from vqvae.lit_vqvae import ( LitVQVAE, ConfigModel_VQVAE, )

def get_modules_and_configs() -> Tuple[Dict[str, pl.LightningModule], List[BaseModel]]:

    registry = { "vqvae":  LitVQVAE }
    configs  = [ConfigModel_VQVAE]

    return registry, configs
