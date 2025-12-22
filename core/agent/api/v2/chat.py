import json
from typing import Annotated, Any, AsyncGenerator, cast

from common.otlp.trace.span import Span
from common.service import get_db_service
from common.service.db.db_service import session_getter
from fastapi import APIRouter, Header
from pydantic import ConfigDict
from starlette.responses import StreamingResponse

from agent.api.schemas_v2.bot_chat_inputs import Chat
from agent.api.schemas_v2.bot_dsl import BotDsl
from agent.api.v1.base_api import CompletionBase
from agent.domain.models.bot import BotRelease
from agent.exceptions.agent_exc import AgentInternalExc
from agent.infra.app_auth import APPAuth
from agent.service.builder.chat_builder import ChatRunnerBuilder
from agent.service.runner.debug_chat_runner import DebugChatRunner

chat_router = APIRouter()

headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class CustomChatCompletion(CompletionBase):
    """Custom chat completion for debug chat agents."""

    bot_id: str
    uid: str
    question: str
    dsl: BotDsl
    model_config = ConfigDict(arbitrary_types_allowed=True)
    span: Span

    def __init__(self, inputs: Chat, **data: Any) -> None:
        super().__init__(inputs=inputs, **data)

    async def build_runner(self, span: Span) -> DebugChatRunner:
        """Build ChatRunnerRunner"""
        builder = ChatRunnerBuilder(
            app_id=self.app_id,
            uid=self.uid,
            span=span,
            inputs=cast(Chat, self.inputs),
            dsl=self.dsl,
        )
        return await builder.build()

    async def do_complete(self) -> AsyncGenerator[str, None]:
        """Run agent"""
        with self.span.start("ChatNode") as sp:
            sp.set_attributes(
                attributes={
                    "app_id": self.app_id,
                    "bot_id": self.bot_id,
                    "uid": self.uid,
                }
            )
            sp.add_info_events(
                {"chat-inputs": self.inputs.model_dump_json(by_alias=True)}
            )
            node_trace = await self.build_node_trace(bot_id=self.bot_id, span=sp)
            meter = await self.build_meter(sp)

            # Use parent class run_runner method which includes _finalize_run logic
            async for response in self.run_runner(node_trace, meter, span=sp):
                yield response


async def _validate_app_auth(app_id: str, sp: Span) -> None:
    app_detail = await APPAuth().app_detail(app_id)
    sp.add_info_events({"kong-app-detail": json.dumps(app_detail, ensure_ascii=False)})
    if app_detail is None:
        raise AgentInternalExc(f"Cannot find appid:{app_id} authentication information")
    if app_detail.get("code") != 0:
        raise AgentInternalExc(app_detail.get("message", ""))
    if len(app_detail.get("data", [])) == 0:
        raise AgentInternalExc(f"Cannot find appid:{app_id} authentication information")


@chat_router.post(  # type: ignore[misc]
    "/bot/chat",
    description="Agent execution - user mode",
    response_model=None,
)
async def bot_chat(
    x_consumer_username: Annotated[str, Header()],
    inputs: Chat,
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
    with span.start("BotChat") as sp:
        sp.set_attribute("bot_id", inputs.bot_id)
        sp.add_info_events({"bot-chat-inputs": inputs.model_dump_json(by_alias=True)})

        await _validate_app_auth(x_consumer_username, sp)
        with session_getter(get_db_service()) as session:
            # Query Bot table data using bot_id
            bot_release = (
                session.query(BotRelease)
                .filter(
                    BotRelease.bot_id == inputs.bot_id,
                    BotRelease.version == inputs.version_name,
                )
                .first()
            )
            if not bot_release:
                raise AgentInternalExc(
                    f"Bot release with bot_id:{inputs.bot_id} version_name:{inputs.version_name} not found"
                )

            completion = CustomChatCompletion(
                app_id=x_consumer_username,
                inputs=inputs,
                log_caller=inputs.meta_data.caller,
                span=span,
                bot_id="",
                uid=inputs.uid,
                question=inputs.get_last_message_content(),
                dsl=BotDsl(**json.loads(bot_release.dsl)),
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
