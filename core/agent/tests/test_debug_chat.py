"""Test debug_chat API endpoints in debug_chat module"""

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.otlp import sid as sid_module
from common.otlp.trace.span import Span
from starlette.responses import StreamingResponse

from agent.api.schemas_v2.bot_chat_inputs import DebugChat, MessageInputs
from agent.api.v2.debug_chat import (
    CustomChatCompletion,
    _validate_tenant,
    bot_debug_chat,
)
from agent.domain.models.bot import Bot, BotTenant
from agent.exceptions.agent_exc import AgentExc


@dataclass
class _DummySidGen:
    """Simple sid generator for testing environment."""

    value: str = "test-sid"

    def gen(self) -> str:  # pragma: no cover - only for testing environment
        return self.value


@pytest.fixture(autouse=True)
def _setup_test_environment() -> None:
    """Automatically inject environment fixes for all tests.

    - Ensure `sid_generator2` is initialized to avoid `Span` construction failure.
    """
    # Initialize sid generator to avoid Span throwing "sid_generator2 is not initialized"
    if sid_module.sid_generator2 is None:
        sid_module.sid_generator2 = _DummySidGen()  # type: ignore[assignment]


class TestCustomChatCompletion:
    """Test CustomChatCompletion class"""

    @pytest.fixture
    def debug_chat_inputs(self) -> DebugChat:
        """Create DebugChat input instance for testing"""
        from agent.api.schemas_v2.bot_dsl import BotDsl, ModelInputs, ModelPropertiesInputs

        dsl = BotDsl(
            name="test_bot",
            model=ModelInputs(
                name="test_model",
                type="openai",
                properties=ModelPropertiesInputs(
                    id="test_model",
                    url="https://api.test.com",
                    token="test_token",
                ),
            ),
        )
        return DebugChat(
            bot_id="test_bot_id",
            conversation_id="test_conversation_id",
            stream=True,
            messages=[
                MessageInputs(content="test question", content_type="text", role="user")
            ],
            dsl=dsl,
        )

    @pytest.fixture
    def span(self) -> Span:
        """Create Span instance for testing"""
        return Span(app_id="test_app", uid="test_uid")

    @pytest.fixture
    def completion(
        self, debug_chat_inputs: DebugChat, span: Span
    ) -> CustomChatCompletion:
        """Create CustomChatCompletion instance for testing"""
        return CustomChatCompletion(
            app_id="test_app",
            inputs=debug_chat_inputs,
            log_caller="test_caller",
            span=span,
            bot_id="test_bot_id",
            uid="test_uid",
            question="test question",
        )

    @pytest.mark.asyncio
    async def test_build_runner(
        self, completion: CustomChatCompletion, span: Span
    ) -> None:
        """Test building DebugChatRunner"""
        mock_runner = AsyncMock()
        mock_builder = AsyncMock()
        mock_builder.build.return_value = mock_runner

        with patch(
            "agent.api.v2.debug_chat.DebugChatRunnerBuilder",
            return_value=mock_builder,
        ):
            runner = await completion.build_runner(span)
            assert runner is not None
            mock_builder.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_complete(self, completion: CustomChatCompletion) -> None:
        """Test executing completion flow"""
        mock_runner = AsyncMock()
        mock_chunk = MagicMock()
        mock_chunk.id = "test_id"
        mock_chunk.object = "chat.completion.chunk"
        mock_chunk.model_dump_json.return_value = '{"test": "data"}'

        async def mock_run() -> AsyncIterator[Any]:
            yield mock_chunk

        mock_runner.run = AsyncMock(return_value=mock_run())

        with patch.object(
            CustomChatCompletion, "build_runner", return_value=mock_runner
        ):
            with patch.object(
                CustomChatCompletion, "build_node_trace", return_value=MagicMock()
            ):
                with patch.object(
                    CustomChatCompletion, "build_meter", return_value=MagicMock()
                ):
                    results = []
                    async for result in completion.do_complete():
                        results.append(result)

                    assert len(results) > 0


class TestValidateTenant:
    """Test _validate_tenant function"""

    @pytest.mark.asyncio
    async def test_validate_tenant_success(self) -> None:
        """Test successful tenant validation"""
        from datetime import datetime

        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_username",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )

        with patch("agent.api.v2.debug_chat.get_db_service"):
            with patch("agent.api.v2.debug_chat.session_getter") as mock_session_getter:
                mock_session = MagicMock()
                mock_query = MagicMock()
                mock_filter = MagicMock()
                mock_query.filter.return_value = mock_filter
                mock_filter.first.return_value = mock_tenant
                mock_session.query.return_value = mock_query
                mock_session_getter.return_value.__enter__.return_value = mock_session
                mock_session_getter.return_value.__exit__.return_value = False

                await _validate_tenant("test_username")

                mock_session.query.assert_called_once_with(BotTenant)
                mock_query.filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_tenant_not_found(self) -> None:
        """Test tenant validation when tenant is not found"""
        with patch("agent.api.v2.debug_chat.get_db_service"):
            with patch("agent.api.v2.debug_chat.session_getter") as mock_session_getter:
                mock_session = MagicMock()
                mock_query = MagicMock()
                mock_filter = MagicMock()
                mock_query.filter.return_value = mock_filter
                mock_filter.first.return_value = None
                mock_session.query.return_value = mock_query
                mock_session_getter.return_value.__enter__.return_value = mock_session
                mock_session_getter.return_value.__exit__.return_value = False

                with pytest.raises(AgentExc) as exc_info:
                    await _validate_tenant("test_username")

                assert exc_info.value.c == 40500
                assert "Cannot find tenant id:test_username" in str(exc_info.value)


class TestBotDebugChatEndpoint:
    """Test bot_debug_chat endpoint"""

    @pytest.fixture
    def debug_chat_inputs(self) -> DebugChat:
        """Create DebugChat input instance for testing"""
        return DebugChat(
            bot_id="test_bot_id",
            conversation_id="test_conversation_id",
            stream=True,
            messages=[
                MessageInputs(content="test question", content_type="text", role="user")
            ],
            dsl=None,
        )

    @pytest.fixture
    def mock_bot(self) -> Bot:
        """Create mock Bot instance"""
        from datetime import datetime

        bot = Bot(
            id="test_bot_id",
            app_id="test_app",
            dsl=json.dumps(
                {
                    "name": "test_bot",
                    "model": {
                        "name": "test_model",
                        "type": "openai",
                        "properties": {
                            "id": "test_model",
                            "url": "https://api.test.com",
                            "token": "test_token",
                        },
                    },
                    "plugin": {},
                    "rag": {
                        "call_mode": "auto",
                        "query_rewrite": False,
                        "query_rewrite_prompt": "",
                    },
                }
            ),
            pub_status=1,
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        return bot

    @pytest.fixture
    def mock_tenant_session(self) -> MagicMock:
        """Create mock database session for tenant validation"""
        from datetime import datetime

        session = MagicMock()
        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_app",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        # Setup query chain mock for tenant
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_tenant
        session.query.return_value = mock_query
        return session

    @pytest.mark.asyncio
    async def test_bot_debug_chat_success(
        self,
        debug_chat_inputs: DebugChat,
        mock_bot: Bot,
        mock_tenant_session: MagicMock,
    ) -> None:
        """Test successful bot debug chat endpoint"""
        mock_completion = AsyncMock()

        async def mock_do_complete() -> AsyncIterator[str]:
            yield 'data: {"test": "chunk"}\n\n'
            yield "data: [DONE]\n\n"

        mock_completion.do_complete = mock_do_complete

        # Mock tenant validation
        with patch("agent.api.v2.debug_chat._validate_tenant") as mock_validate:
            mock_validate.return_value = None

            # Mock database session
            with patch("agent.api.v2.debug_chat.get_db_service"):
                with patch(
                    "agent.api.v2.debug_chat.session_getter"
                ) as mock_session_getter:
                    # Setup session for bot query
                    mock_bot_session = MagicMock()
                    mock_bot_query = MagicMock()
                    mock_bot_filter = MagicMock()
                    mock_bot_filter.first.return_value = mock_bot
                    mock_bot_query.filter.return_value = mock_bot_filter
                    mock_bot_session.query.return_value = mock_bot_query

                    # Return bot session (tenant validation is mocked, so no tenant session needed)
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_bot_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    # Mock CustomChatCompletion
                    with patch(
                        "agent.api.v2.debug_chat.CustomChatCompletion",
                        return_value=mock_completion,
                    ):
                        response = await bot_debug_chat(
                            x_consumer_username="test_app",
                            inputs=debug_chat_inputs,
                        )

                        assert isinstance(response, StreamingResponse)
                        assert response.media_type == "text/event-stream"

                        # Verify response content
                        content = b""
                        async for chunk in response.body_iterator:
                            if isinstance(chunk, bytes):
                                content += chunk
                            elif isinstance(chunk, str):
                                content += chunk.encode("utf-8")
                            else:
                                content += bytes(chunk)

                        assert b"[DONE]" in content

                        # Verify dsl was set from bot
                        assert debug_chat_inputs.dsl is not None

    @pytest.mark.asyncio
    async def test_bot_debug_chat_with_dsl_provided(
        self,
        mock_bot: Bot,
        mock_tenant_session: MagicMock,
    ) -> None:
        """Test bot debug chat endpoint when dsl is already provided"""
        from agent.api.schemas_v2.bot_dsl import BotDsl, ModelInputs, ModelPropertiesInputs

        provided_dsl = BotDsl(
            name="provided_bot",
            model=ModelInputs(
                name="provided_model",
                type="openai",
                properties=ModelPropertiesInputs(
                    id="provided_model",
                    url="https://api.provided.com",
                    token="provided_token",
                ),
            ),
        )
        debug_chat_inputs = DebugChat(
            bot_id="test_bot_id",
            conversation_id="test_conversation_id",
            stream=True,
            messages=[
                MessageInputs(content="test question", content_type="text", role="user")
            ],
            dsl=provided_dsl,
        )

        mock_completion = AsyncMock()

        async def mock_do_complete() -> AsyncIterator[str]:
            yield "data: [DONE]\n\n"

        mock_completion.do_complete = mock_do_complete

        # Mock tenant validation
        with patch("agent.api.v2.debug_chat._validate_tenant") as mock_validate:
            mock_validate.return_value = None

            # Mock database session
            with patch("agent.api.v2.debug_chat.get_db_service"):
                with patch(
                    "agent.api.v2.debug_chat.session_getter"
                ) as mock_session_getter:
                    # Setup session for bot query
                    mock_bot_session = MagicMock()
                    mock_bot_query = MagicMock()
                    mock_bot_filter = MagicMock()
                    mock_bot_filter.first.return_value = mock_bot
                    mock_bot_query.filter.return_value = mock_bot_filter
                    mock_bot_session.query.return_value = mock_bot_query

                    # Return bot session (tenant validation is mocked, so no tenant session needed)
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_bot_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    # Mock CustomChatCompletion
                    with patch(
                        "agent.api.v2.debug_chat.CustomChatCompletion",
                        return_value=mock_completion,
                    ):
                        response = await bot_debug_chat(
                            x_consumer_username="test_app",
                            inputs=debug_chat_inputs,
                        )

                        assert isinstance(response, StreamingResponse)
                        # Verify dsl was not changed (still the provided one)
                        assert debug_chat_inputs.dsl == provided_dsl

    @pytest.mark.asyncio
    async def test_bot_debug_chat_bot_not_found(
        self,
        debug_chat_inputs: DebugChat,
        mock_tenant_session: MagicMock,
    ) -> None:
        """Test bot debug chat endpoint when bot is not found"""
        # Mock tenant validation
        with patch("agent.api.v2.debug_chat._validate_tenant") as mock_validate:
            mock_validate.return_value = None

            # Mock database session
            with patch("agent.api.v2.debug_chat.get_db_service"):
                with patch(
                    "agent.api.v2.debug_chat.session_getter"
                ) as mock_session_getter:
                    # Setup session for bot query
                    mock_bot_session = MagicMock()
                    mock_bot_query = MagicMock()
                    mock_bot_filter = MagicMock()
                    mock_bot_filter.first.return_value = None
                    mock_bot_query.filter.return_value = mock_bot_filter
                    mock_bot_session.query.return_value = mock_bot_query

                    # Return bot session (tenant validation is mocked, so no tenant session needed)
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_bot_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    with pytest.raises(AgentExc) as exc_info:
                        await bot_debug_chat(
                            x_consumer_username="test_app",
                            inputs=debug_chat_inputs,
                        )

                    assert exc_info.value.c == 40500
                    assert "Bot with id:test_bot_id not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bot_debug_chat_tenant_validation_failed(
        self, debug_chat_inputs: DebugChat
    ) -> None:
        """Test bot debug chat endpoint when tenant validation fails"""
        from agent.exceptions.agent_exc import AgentExc, AgentInternalExc

        # Mock tenant validation to raise exception
        with patch("agent.api.v2.debug_chat._validate_tenant") as mock_validate:
            mock_validate.side_effect = AgentInternalExc

            with pytest.raises(AgentExc) as exc_info:
                await bot_debug_chat(
                    x_consumer_username="test_app",
                    inputs=debug_chat_inputs,
                )

            assert exc_info.value.c == 40500
