import json

from agent.api.schemas_v2.bot_chat_inputs import DebugChat
from typing import Annotated, Any, AsyncGenerator, cast
from agent.api.schemas_v2.bot_dsl import Dsl

from fastapi import APIRouter, Header
from pydantic import ConfigDict
from starlette.responses import StreamingResponse
from common.otlp.trace.span import Span
from common.service import get_db_service
from common.service.db.db_service import session_getter
from agent.domain.models.bot import Bot, BotTenant
from agent.exceptions.bot_exc import (
    BotNotFoundExc,
    TenantNotFoundExc,
)
from agent.exceptions.agent_exc import AgentInternalExc

from agent.api.v1.base_api import CompletionBase
from agent.service.builder.debug_chat_builder import DebugChatRunnerBuilder
from agent.service.runner.debug_chat_runner import DebugChatRunner


debug_chat_router = APIRouter()

headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class CustomChatCompletion(CompletionBase):
    """Custom chat completion for debug chat agents."""
    bot_id: str
    uid: str
    question: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
    span: Span

    def __init__(self, inputs: DebugChat, **data: Any) -> None:
        super().__init__(inputs=inputs, **data)

    async def build_runner(self, span: Span) -> DebugChatRunner:
        """Build DebugChatRunnerRunner"""
        builder = DebugChatRunnerBuilder(
            app_id=self.app_id,
            uid=self.uid,
            span=span,
            inputs=cast(DebugChat, self.inputs),
        )
        return await builder.build()

    async def do_complete(self) -> AsyncGenerator[str, None]:
        """Run agent"""
        with self.span.start("DebugChatNode") as sp:
            sp.set_attributes(
                attributes={
                    "app_id": self.app_id,
                    "bot_id": self.bot_id,
                    "uid": self.uid,
                }
            )
            sp.add_info_events({"debug-chat-inputs": self.inputs.model_dump_json(by_alias=True)})
            node_trace = await self.build_node_trace(bot_id=self.bot_id, span=sp)
            meter = await self.build_meter(sp)

            # Use parent class run_runner method which includes _finalize_run logic
            async for response in self.run_runner(node_trace, meter, span=sp):
                yield response


async def _validate_tenant(x_consumer_username: str) -> None:
    """Validate tenant information"""
    with session_getter(get_db_service()) as session:
        # Query if the record exists by x_consumer_username
        existing_tenant = (
            session.query(BotTenant)
            .filter(BotTenant.alias_id == x_consumer_username)
            .first()
        )
        if not existing_tenant:
            raise AgentInternalExc(
                f"Cannot find tenant id:{x_consumer_username} information"
            )


@debug_chat_router.post(  # type: ignore[misc]
    "/bot/debug_chat",
    description="Agent execution - user mode",
    response_model=None,
)
async def bot_debug_chat(
    x_consumer_username: Annotated[str, Header()],
    inputs: DebugChat,
) -> StreamingResponse:
    """Agent execution - user mode

    Args:
        completion_inputs: Request body
        app_id: Application ID
        bot_id: Bot ID
        uid: User ID
        span: Trace object

    Returns:
        Streaming response
    """

    span = Span(app_id=x_consumer_username)
    with span.start("BotDebugChat") as sp:
        sp.set_attribute("bot_id", inputs.bot_id)
        sp.add_info_events(
            {"bot-debug-chat-inputs": inputs.model_dump_json(by_alias=True)}
        )

        await _validate_tenant(x_consumer_username)
        with session_getter(get_db_service()) as session:
            # Query Bot table data using bot_id
            bot = session.query(Bot).filter(Bot.id == inputs.bot_id).first()
            if not bot:
                raise AgentInternalExc(f"Bot with id:{inputs.bot_id} not found")

            if inputs.dsl is None:
                inputs.dsl = Dsl(**json.loads(bot.dsl))

            completion = CustomChatCompletion(
                app_id=x_consumer_username,
                inputs=inputs,
                log_caller=inputs.meta_data.caller,
                span=span,
                bot_id="",
                uid=inputs.uid,
                question=inputs.get_last_message_content(),
            )

            async def generate() -> AsyncGenerator[str, None]:
                """Generator for streaming response."""
                async for response in completion.do_complete():
                    # Convert chunk to JSON string for streaming response
                    yield response

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers=headers,
            )
