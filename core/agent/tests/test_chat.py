"""Test chat API endpoints in chat module"""

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.otlp import sid as sid_module
from common.otlp.trace.span import Span
from starlette.responses import StreamingResponse

from agent.api.schemas_v2.bot_chat_inputs import Chat, MessageInputs
from agent.api.v2.chat import CustomChatCompletion, _validate_app_auth, bot_chat
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
    def chat_inputs(self) -> Chat:
        """Create Chat input instance for testing"""
        return Chat(
            bot_id="test_bot_id",
            uid="test_uid",
            conversation_id="test_conversation_id",
            stream=True,
            messages=[
                MessageInputs(content="test question", content_type="text", role="user")
            ],
            version_name="v1.0.0",
        )

    @pytest.fixture
    def span(self) -> Span:
        """Create Span instance for testing"""
        return Span(app_id="test_app", uid="test_uid")

    @pytest.fixture
    def completion(self, chat_inputs: Chat, span: Span) -> CustomChatCompletion:
        """Create CustomChatCompletion instance for testing"""
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
        return CustomChatCompletion(
            app_id="test_app",
            inputs=chat_inputs,
            log_caller="test_caller",
            span=span,
            bot_id="test_bot_id",
            uid="test_uid",
            question="test question",
            dsl=dsl,
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
            "agent.api.v2.chat.ChatRunnerBuilder",
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


class TestValidateAppAuth:
    """Test _validate_app_auth function"""

    @pytest.mark.asyncio
    async def test_validate_app_auth_success(self) -> None:
        """Test successful app authentication validation"""
        mock_span = MagicMock()
        mock_app_detail = {"code": 0, "data": [{"id": "test_app"}]}

        with patch("agent.api.v2.chat.APPAuth") as mock_app_auth_class:
            mock_app_auth = AsyncMock()
            mock_app_auth.app_detail.return_value = mock_app_detail
            mock_app_auth_class.return_value = mock_app_auth

            await _validate_app_auth("test_app", mock_span)

            mock_app_auth.app_detail.assert_called_once_with("test_app")

    @pytest.mark.asyncio
    async def test_validate_app_auth_not_found(self) -> None:
        """Test app authentication validation when app is not found"""
        mock_span = MagicMock()

        with patch("agent.api.v2.chat.APPAuth") as mock_app_auth_class:
            mock_app_auth = AsyncMock()
            mock_app_auth.app_detail.return_value = None
            mock_app_auth_class.return_value = mock_app_auth

            with pytest.raises(AgentExc) as exc_info:
                await _validate_app_auth("test_app", mock_span)

            assert exc_info.value.c == 40500
            assert "Cannot find appid:test_app" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_app_auth_code_not_zero(self) -> None:
        """Test app authentication validation when code is not zero"""
        mock_span = MagicMock()
        mock_app_detail = {"code": 1, "message": "Authentication failed"}

        with patch("agent.api.v2.chat.APPAuth") as mock_app_auth_class:
            mock_app_auth = AsyncMock()
            mock_app_auth.app_detail.return_value = mock_app_detail
            mock_app_auth_class.return_value = mock_app_auth

            with pytest.raises(AgentExc) as exc_info:
                await _validate_app_auth("test_app", mock_span)

            assert exc_info.value.c == 40500
            assert "Authentication failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_app_auth_empty_data(self) -> None:
        """Test app authentication validation when data is empty"""
        mock_span = MagicMock()
        mock_app_detail = {"code": 0, "data": []}

        with patch("agent.api.v2.chat.APPAuth") as mock_app_auth_class:
            mock_app_auth = AsyncMock()
            mock_app_auth.app_detail.return_value = mock_app_detail
            mock_app_auth_class.return_value = mock_app_auth

            with pytest.raises(AgentExc) as exc_info:
                await _validate_app_auth("test_app", mock_span)

            assert exc_info.value.c == 40500
            assert "Cannot find appid:test_app" in str(exc_info.value)


class TestBotChatEndpoint:
    """Test bot_chat endpoint"""

    @pytest.fixture
    def chat_inputs(self) -> Chat:
        """Create Chat input instance for testing"""
        return Chat(
            bot_id="test_bot_id",
            uid="test_uid",
            conversation_id="test_conversation_id",
            stream=True,
            messages=[
                MessageInputs(content="test question", content_type="text", role="user")
            ],
            version_name="v1.0.0",
        )

    @pytest.fixture
    def mock_bot_release(self) -> MagicMock:
        """Create mock BotRelease instance"""
        mock_release = MagicMock()
        mock_release.bot_id = "test_bot_id"
        mock_release.version = "v1.0.0"
        mock_release.dsl = json.dumps(
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
            }
        )
        return mock_release

    @pytest.mark.asyncio
    async def test_bot_chat_success(
        self, chat_inputs: Chat, mock_bot_release: MagicMock
    ) -> None:
        """Test successful bot chat endpoint"""
        mock_completion = AsyncMock()

        async def mock_do_complete() -> AsyncIterator[str]:
            yield 'data: {"test": "chunk"}\n\n'
            yield "data: [DONE]\n\n"

        mock_completion.do_complete = mock_do_complete

        # Mock app auth validation
        with patch("agent.api.v2.chat._validate_app_auth") as mock_validate:
            mock_validate.return_value = None

            # Mock database session
            with patch("agent.api.v2.chat.get_db_service"):
                with patch("agent.api.v2.chat.session_getter") as mock_session_getter:
                    mock_session = MagicMock()
                    mock_query = MagicMock()
                    mock_filter = MagicMock()
                    mock_filter.first.return_value = mock_bot_release
                    mock_query.filter.return_value = mock_filter
                    mock_session.query.return_value = mock_query
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    # Mock CustomChatCompletion
                    with patch(
                        "agent.api.v2.chat.CustomChatCompletion",
                        return_value=mock_completion,
                    ):
                        response = await bot_chat(
                            x_consumer_username="test_app",
                            inputs=chat_inputs,
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

    @pytest.mark.asyncio
    async def test_bot_chat_bot_release_not_found(self, chat_inputs: Chat) -> None:
        """Test bot chat endpoint when bot release is not found"""
        # Mock app auth validation
        with patch("agent.api.v2.chat._validate_app_auth") as mock_validate:
            mock_validate.return_value = None

            # Mock database session
            with patch("agent.api.v2.chat.get_db_service"):
                with patch("agent.api.v2.chat.session_getter") as mock_session_getter:
                    mock_session = MagicMock()
                    mock_query = MagicMock()
                    mock_filter = MagicMock()
                    mock_filter.first.return_value = None
                    mock_query.filter.return_value = mock_filter
                    mock_session.query.return_value = mock_query
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    with pytest.raises(AgentExc) as exc_info:
                        await bot_chat(
                            x_consumer_username="test_app",
                            inputs=chat_inputs,
                        )

                    assert exc_info.value.c == 40500
                    assert "Bot release" in str(exc_info.value)
                    assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bot_chat_app_auth_failed(self, chat_inputs: Chat) -> None:
        """Test bot chat endpoint when app authentication fails"""
        from agent.exceptions.agent_exc import AgentExc, AgentInternalExc

        # Mock app auth validation to raise exception
        with patch("agent.api.v2.chat._validate_app_auth") as mock_validate:
            mock_validate.side_effect = AgentInternalExc

            with pytest.raises(AgentExc) as exc_info:
                await bot_chat(
                    x_consumer_username="test_app",
                    inputs=chat_inputs,
                )

            assert exc_info.value.c == 40500
