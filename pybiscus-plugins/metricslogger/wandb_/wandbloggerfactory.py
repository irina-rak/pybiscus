
from enum import Enum
import os
from typing import ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict

import wandb

from pybiscus.interfaces.core.metricsloggerfactory import MetricsLoggerFactory
from pybiscus.core.pybiscus_logger import console

class EnvVar(BaseModel):

    env_var_name: str = "WANDB_API_KEY"    

    model_config = ConfigDict(extra="forbid")

class String(BaseModel):

    value: str = "xxxxxxxxxx"    

    model_config = ConfigDict(extra="forbid")

class Undefined(BaseModel):

    model_config = ConfigDict(extra="forbid")
    
class WandbRole(Enum):
    Client = True
    Server = False

class ConfigWandbLoggerFactoryParams(BaseModel):
    """
        project: Name of the W&B project
        entity: Optional name of the W&B entity (team or username)
        name: Optional name for this specific run
        group: Optional group to organize related runs
        job_type: Optional job type (e.g. 'server', 'client')
        config: Optional dictionary of configuration parameters to log
        is_client: Boolean indicating if this is a client
        partition_id: Optional client partition ID (required if is_client=True)
    """

    PYBISCUS_CONFIG: ClassVar[str] = "config"

    project: str = "DefaultProjectName"
    # entity:  str = "DefaultEntityName"
    group:   str = "DefaultRunGroup"
    name:    str = "DefaultRunName"
    job_type: str = "DefaultJobType"
    # config: Optional[Dict[str, Any]] = None,
    #     config={
    #    "round": config["round"],
    # )
    # config={
    #     "model_checkpoint": model_checkpoint,
    #     "num_rounds": num_rounds,
    #     "fraction_fit": fraction_fit,
    #     "num_labels": num_labels,
    # },

    is_client: WandbRole = WandbRole.Server.value
    # partition_id: int  = 1

    model_config = ConfigDict(extra="forbid")

class ConfigWandbLoggerFactoryData(BaseModel):

    PYBISCUS_ALIAS: ClassVar[str] = """🚨 <span style="color: white; background-color: red;">Untested Wandb configuration</span> 🚨"""


    api_key_definition: Union[EnvVar, String, Undefined]
    params:             ConfigWandbLoggerFactoryParams

    model_config = ConfigDict(extra="forbid")

class ConfigWandbLoggerFactory(BaseModel):

    name:   Literal["wandb"]

    PYBISCUS_ALIAS: ClassVar[str] = "WanDb"

    config: ConfigWandbLoggerFactoryData

    model_config = ConfigDict(extra="forbid")

    # to emulate a dict
    def __getitem__(self, attName):
        return getattr(self, attName, None)


class WandbLoggerFactory(MetricsLoggerFactory):

    def __init__(self, config=None, **kwargs):
        # Accept config as a keyword argument for compatibility
        self.conf = config if config is not None else kwargs.get('conf')

    # def get_metricslogger(self, reporting_path):
    #     if wandb.run is not None:
    #         return wandb.run

    #     if isinstance(self.conf.api_key_definition, EnvVar):
    #         print("C'est une EnvVar")
    #         WANDB_API_KEY = os.getenv(self.conf.api_key_definition.env_var_name)
    #         if not WANDB_API_KEY:
    #             raise RuntimeError(f"WANDB_API_KEY environment variable '{self.conf.api_key_definition.env_var_name}' is not set.")
    #         os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    #         wandb.init()
    #     elif isinstance(self.conf.api_key_definition, String):
    #         print("C'est une String")
    #         wandb.init(self.conf.api_key_definition.value)
    #     elif isinstance(self.conf.api_key_definition, Undefined):
    #         print("C'est un Undefined")
    #         wandb.init()
    #     else:
    #         return None

    #     # Only pass allowed keys to wandb.init
    #     params = self.conf.params.model_dump()
    #     allowed_keys = {"project", "entity", "group", "name", "job_type", "config"}
    #     wandb_params = {k: v for k, v in params.items() if k in allowed_keys and v is not None}
    #     wandb_server_run = wandb.init(**wandb_params)

    #     if wandb_server_run is None:
    #         return None
    #     else:
    #         return wandb_server_run

    def get_metricslogger(self, reporting_path):
        if wandb.run is not None:
            return wandb.run
    
        # Set API key in environment if needed
        if isinstance(self.conf.api_key_definition, EnvVar):
            WANDB_API_KEY = os.getenv(self.conf.api_key_definition.env_var_name)
            if not WANDB_API_KEY:
                raise RuntimeError(f"WANDB_API_KEY environment variable '{self.conf.api_key_definition.env_var_name}' is not set.")
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        elif isinstance(self.conf.api_key_definition, String):
            os.environ["WANDB_API_KEY"] = self.conf.api_key_definition.value
        # Undefined: do nothing, rely on existing env
    
        # Only pass allowed keys to wandb.init
        params = self.conf.params.model_dump()
        allowed_keys = {"project", "entity", "group", "name", "job_type", "config"}
        wandb_params = {k: v for k, v in params.items() if k in allowed_keys and v is not None}
        wandb_server_run = wandb.init(**wandb_params)
    
        if wandb_server_run is None:
            return None
        else:
            return wandb_server_run
        