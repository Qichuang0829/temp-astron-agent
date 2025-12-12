from typing import Optional

from pydantic import BaseModel, Field
from agent.api.schemas_v2.bot_dsl import Dsl


class ProtocolSynchronization(BaseModel):
    id: Optional[str] = Field(default=None)
    dsl: Dsl = Field(...)


class Publish(BaseModel):
    bot_id: str = Field(...)
    version: str = Field(...)
    description: str = Field(...)
    dsl: Optional[Dsl] = Field(default=None)


class Auth(BaseModel):
    version_id: int = Field(...)
    app_id: str = Field(...)
