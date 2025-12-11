from typing import Optional

from pydantic import BaseModel, Field

from agent.exceptions.bot_exc import BotExc


class BotResponse(BaseModel):
    code: int = Field(default=0)
    """Bot config management response status code"""

    message: str = Field(default="success")
    """Bot config management status code description message"""

    data: Optional[dict] = Field(default=None)
    """Bot config data"""


def build_bot_response(
    error: BotExc, data: Optional[dict] = None, message: Optional[str] = None
) -> BotResponse:
    """构建 Bot 管理响应（成功或失败）

    Args:
        error: 错误对象（BotExc(*c_0) 表示成功）
        data: 返回的数据，成功时传入，失败时为 None
        message: 可选的自定义消息，如果提供则覆盖 error.m

    Returns:
        BotResponse: 响应对象
    """

    return BotResponse(code=error.c, message=message or error.m, data=data)
