"""Workflow Agent API endpoints."""

from typing import Annotated, AsyncGenerator, cast

from common.otlp.trace.span import Span
from fastapi import APIRouter, Header
from starlette.responses import StreamingResponse

from agent.api.schemas.workflow_agent_inputs import CustomCompletionInputs

debug_chat_router = APIRouter()

headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

@debug_chat_router.post(  # type: ignore[misc]
    "/bot/debug_chat",
    description="bot debug",
    response_model=None,
)
async def bot_debug_chat(
    x_consumer_username: Annotated[str, Header()],
    bot_debug_chat_inputs: CustomCompletionInputs,
) -> StreamingResponse:


    span = Span(app_id=x_consumer_username, uid="")
    
    completion = CustomChatCompletion(
        app_id=x_consumer_username,
        inputs=completion_inputs,
        log_caller=completion_inputs.meta_data.caller,
        span=span,
        bot_id="",
        uid=completion_inputs.uid,
        question=completion_inputs.get_last_message_content(),
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
