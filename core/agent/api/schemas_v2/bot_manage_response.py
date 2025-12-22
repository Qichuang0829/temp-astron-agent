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
    """Build Bot management response (success or failure)

    Args:
        error: Error object (BotExc(*c_0) indicates success)
        data: Returned data, passed in on success, None on failure
        message: Optional custom message, if provided, overrides error.m

    Returns:
        BotResponse: Response object
    """

    return BotResponse(code=error.c, message=message or error.m, data=data or None)
