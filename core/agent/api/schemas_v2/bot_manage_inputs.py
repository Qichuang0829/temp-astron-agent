from typing import Optional

from pydantic import BaseModel, Field
from agent.api.schemas_v2.bot_dsl import Dsl as DslInputs


class ProtocolSynchronization(BaseModel):
    id: Optional[str] = Field(default=None)
    dsl: DslInputs = Field(...)


class Publish(BaseModel):
    bot_id: str = Field(...)
    version: str = Field(...)
    description: str = Field(...)
    dsl: Optional[DslInputs] = Field(default=None)


class Auth(BaseModel):
    version_id: int = Field(...)
    app_id: str = Field(...)
