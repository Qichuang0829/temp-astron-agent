from typing import Optional

from pydantic import BaseModel, Field

from agent.api.schemas_v2.bot_dsl import BotDsl


class ProtocolSynchronization(BaseModel):
    id: Optional[int] = Field(default=None)
    dsl: BotDsl = Field(...)


class Publish(BaseModel):
    bot_id: int = Field(...)
    version_name: str = Field(...)
    description: str = Field(default="")


class Auth(BaseModel):
    version_id: int = Field(...)
    app_id: str = Field(...)
