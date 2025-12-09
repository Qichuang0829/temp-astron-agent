from datetime import datetime
from typing import Optional

from common.utils.snowfake import get_id
from sqlmodel import Field, SQLModel


class Bot(SQLModel, table=True):  # type: ignore[valid-type,misc]
    __tablename__ = "bot"

    id: str = Field(
        default_factory=lambda: get_id, primary_key=True, description="主键id、雪花id"
    )
    app_id: str = Field(..., description="租户应用标识")
    dsl: str = Field(..., description="助手编排协议")
    pub_status: int = Field(..., description="助手当前状态：0-草稿、1-已发布")
    create_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    update_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class BotTenant(SQLModel, table=True):  # type: ignore[valid-type,misc]
    __tablename__ = "bot_tenant"

    id: str = Field(
        default_factory=lambda: get_id, primary_key=True, description="主键id、雪花id"
    )
    name: str = Field(..., max_length=64, description="应用名")
    alias_id: str = Field(..., max_length=32, unique=True, description="应用标识id")
    description: Optional[str] = Field(
        default=None, max_length=255, description="租户描述"
    )
    api_key: Optional[str] = Field(
        default=None, max_length=128, description="租户api key"
    )
    api_secret: Optional[str] = Field(
        default=None, max_length=128, description="租户api秘钥"
    )
    create_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    update_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class BotRelease(SQLModel, table=True):  # type: ignore[valid-type,misc]
    __tablename__ = "bot_release"

    id: str = Field(
        default_factory=lambda: get_id, primary_key=True, description="主键id、雪花id"
    )
    bot_id: str = Field(..., description="业务外键、助手表主键")
    version: str = Field(..., max_length=64, description="版本")
    description: Optional[str] = Field(
        default=None, max_length=255, description="版本描述"
    )
    dsl: str = Field(..., description="当前版本助手编排协议")
    create_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    update_at: datetime = Field(default_factory=datetime.now, description="更新时间")
