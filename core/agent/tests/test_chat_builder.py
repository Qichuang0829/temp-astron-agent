"""Test ChatRunnerBuilder class"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.otlp import sid as sid_module
from common.otlp.trace.span import Span

from agent.api.schemas_v2.bot_chat_inputs import Chat, MessageInputs
from agent.api.schemas_v2.bot_dsl import (
    BotDsl,
    KnowledgeInputs,
    ModelInputs,
    ModelPropertiesInputs,
    PropertiesInputs,
)
from agent.service.builder.chat_builder import ChatRunnerBuilder, KnowledgeQueryParams
from agent.service.plugin.base import BasePlugin
from agent.service.runner.debug_chat_runner import DebugChatRunner


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


class TestChatRunnerBuilder:
    """Test ChatRunnerBuilder class"""

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
    def dsl(self) -> BotDsl:
        """Create Dsl instance for testing"""
        from agent.api.schemas_v2.bot_dsl import PluginInputs, RagInputs

        return BotDsl(
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
            plugin=PluginInputs(),
            rag=RagInputs(
                call_mode="auto",
                query_rewrite=False,
                query_rewrite_prompt="",
            ),
        )

    @pytest.fixture
    def span(self) -> Span:
        """Create Span instance for testing"""
        return Span(app_id="test_app", uid="test_uid")

    @pytest.fixture
    def builder(self, chat_inputs: Chat, dsl: BotDsl, span: Span) -> ChatRunnerBuilder:
        """Create Builder instance for testing"""
        return ChatRunnerBuilder(
            app_id="test_app",
            uid="test_uid",
            span=span,
            inputs=chat_inputs,
            dsl=dsl,
        )

    @pytest.mark.asyncio
    async def test_build(self, builder: ChatRunnerBuilder) -> None:
        """Test building DebugChatRunner"""
        mock_model = MagicMock()
        mock_plugins: list[BasePlugin] = []
        from agent.engine.nodes.chat.chat_runner import ChatRunner
        from agent.engine.nodes.cot.cot_runner import CotRunner

        mock_chat_runner = MagicMock(spec=ChatRunner)
        mock_process_runner = MagicMock()
        mock_cot_runner = MagicMock(spec=CotRunner)

        with patch.object(ChatRunnerBuilder, "create_model", return_value=mock_model):
            with patch.object(
                ChatRunnerBuilder, "build_plugins", return_value=mock_plugins
            ):
                with patch.object(
                    ChatRunnerBuilder,
                    "query_knowledge_by_workflow",
                    return_value=([], ""),
                ):
                    with patch.object(
                        ChatRunnerBuilder,
                        "build_chat_runner",
                        return_value=mock_chat_runner,
                    ):
                        with patch.object(
                            ChatRunnerBuilder,
                            "build_process_runner",
                            return_value=mock_process_runner,
                        ):
                            with patch.object(
                                ChatRunnerBuilder,
                                "build_cot_runner",
                                return_value=mock_cot_runner,
                            ):
                                runner = await builder.build()

                                assert isinstance(runner, DebugChatRunner)
                                assert runner.chat_runner == mock_chat_runner
                                assert runner.cot_runner == mock_cot_runner

    @pytest.mark.asyncio
    async def test_build_with_plugins(
        self, builder: ChatRunnerBuilder, dsl: BotDsl
    ) -> None:
        """Test building with plugins from DSL"""
        mock_model = MagicMock()
        mock_plugin = MagicMock(spec=BasePlugin)
        mock_plugins: list[BasePlugin] = [mock_plugin]
        from agent.engine.nodes.chat.chat_runner import ChatRunner
        from agent.engine.nodes.cot.cot_runner import CotRunner

        mock_chat_runner = MagicMock(spec=ChatRunner)
        mock_process_runner = MagicMock()
        mock_cot_runner = MagicMock(spec=CotRunner)

        # Add plugin configuration to DSL
        from agent.api.schemas_v2.bot_dsl import PluginInputs

        plugin_inputs = PluginInputs()
        dsl.plugin = plugin_inputs

        with patch.object(ChatRunnerBuilder, "create_model", return_value=mock_model):
            with patch.object(
                ChatRunnerBuilder, "build_plugins", return_value=mock_plugins
            ):
                with patch.object(
                    ChatRunnerBuilder,
                    "query_knowledge_by_workflow",
                    return_value=([], ""),
                ):
                    with patch.object(
                        ChatRunnerBuilder,
                        "build_chat_runner",
                        return_value=mock_chat_runner,
                    ):
                        with patch.object(
                            ChatRunnerBuilder,
                            "build_process_runner",
                            return_value=mock_process_runner,
                        ):
                            with patch.object(
                                ChatRunnerBuilder,
                                "build_cot_runner",
                                return_value=mock_cot_runner,
                            ):
                                runner = await builder.build()

                                assert isinstance(runner, DebugChatRunner)
                                assert runner.plugins == mock_plugins

    @pytest.mark.asyncio
    async def test_query_knowledge_by_workflow_empty(
        self, builder: ChatRunnerBuilder, span: Span
    ) -> None:
        """Test querying knowledge base (empty list)"""
        metadata_list, backgrounds = await builder.query_knowledge_by_workflow([], span)

        assert metadata_list == []
        assert backgrounds == ""

    @pytest.mark.asyncio
    async def test_query_knowledge_by_workflow_with_knowledge(
        self, builder: ChatRunnerBuilder, span: Span
    ) -> None:
        """Test querying knowledge base (with knowledge base configuration)"""
        knowledge_input = KnowledgeInputs(
            name="test_knowledge",
            description="test description",
            type="AIUI-RAG2",
            properties=PropertiesInputs(
                repos=["repo1"], docs=["doc1"], top_k=3, min_score=0.3
            ),
        )
        # Add repo_type attribute to avoid AttributeError in source code
        object.__setattr__(knowledge_input, "repo_type", None)

        mock_result = {
            "data": {
                "results": [
                    {
                        "title": "Test Doc",
                        "docId": "doc1",
                        "content": "Test content",
                        "references": {},
                    }
                ]
            }
        }

        with patch.object(
            ChatRunnerBuilder,
            "exec_query_knowledge",
            return_value=mock_result,
        ):
            metadata_list, backgrounds = await builder.query_knowledge_by_workflow(
                [knowledge_input], span
            )

            assert len(metadata_list) > 0
            assert backgrounds != ""

    def test_create_knowledge_tasks_no_repo_or_doc_ids(
        self, builder: ChatRunnerBuilder, span: Span
    ) -> None:
        """Test creating knowledge query tasks (no repo_ids and doc_ids)"""
        knowledge_input = KnowledgeInputs(
            name="test",
            description="test",
            type="AIUI-RAG2",
            properties=PropertiesInputs(repos=[], docs=None, top_k=3, min_score=0.3),
        )
        # Add repo_type attribute to avoid AttributeError in source code
        object.__setattr__(knowledge_input, "repo_type", None)

        tasks = builder._create_knowledge_tasks([knowledge_input], span)
        assert len(tasks) == 0

    def test_create_knowledge_tasks_with_repo_ids(
        self, builder: ChatRunnerBuilder, span: Span
    ) -> None:
        """Test creating knowledge query tasks (with repo_ids)"""
        knowledge_input = KnowledgeInputs(
            name="test",
            description="test",
            type="AIUI-RAG2",
            properties=PropertiesInputs(
                repos=["repo1"], docs=None, top_k=3, min_score=0.3
            ),
        )
        # Add repo_type attribute to avoid AttributeError in source code
        object.__setattr__(knowledge_input, "repo_type", None)

        tasks = builder._create_knowledge_tasks([knowledge_input], span)
        assert len(tasks) == 1

    def test_create_knowledge_tasks_with_doc_ids(
        self, builder: ChatRunnerBuilder, span: Span
    ) -> None:
        """Test creating knowledge query tasks (with doc_ids)"""
        knowledge_input = KnowledgeInputs(
            name="test",
            description="test",
            type="CBG-RAG",
            properties=PropertiesInputs(
                repos=["repo1"], docs=["doc1"], top_k=3, min_score=0.3
            ),
        )
        # Add repo_type attribute to avoid AttributeError in source code
        object.__setattr__(knowledge_input, "repo_type", None)

        tasks = builder._create_knowledge_tasks([knowledge_input], span)
        assert len(tasks) == 1

    def test_process_knowledge_results(self, builder: ChatRunnerBuilder) -> None:
        """Test processing knowledge query results"""
        results = [
            {
                "data": {
                    "results": [
                        {
                            "title": "Doc 1",
                            "docId": "doc1",
                            "content": "Content 1",
                            "references": {},
                        },
                        {
                            "title": "Doc 2",
                            "docId": "doc1",
                            "content": "Content 2",
                            "references": {},
                        },
                    ]
                }
            }
        ]

        metadata_list, metadata_map = builder._process_knowledge_results(results)

        assert len(metadata_list) > 0
        assert "doc1" in metadata_map
        assert len(metadata_map["doc1"]) == 2

    def test_process_content_references_image(self, builder: ChatRunnerBuilder) -> None:
        """Test processing content references (image)"""
        content = "See <ref1> for details"
        references = {
            "ref1": {"format": "image", "link": "http://example.com/image.jpg"}
        }

        result = builder._process_content_references(content, references)
        assert "![alt](http://example.com/image.jpg)" in result

    def test_process_content_references_table(self, builder: ChatRunnerBuilder) -> None:
        """Test processing content references (table)"""
        content = "Table: <ref1>"
        references = {
            "ref1": {"format": "table", "content": "|col1|col2|\n|val1|val2|"}
        }

        result = builder._process_content_references(content, references)
        assert "|col1|col2|" in result

    def test_process_content_references_string(
        self, builder: ChatRunnerBuilder
    ) -> None:
        """Test processing content references (string)"""
        content = "See {ref1} for details"
        references = {"ref1": "http://example.com/image.jpg"}

        result = builder._process_content_references(content, references)
        assert "![alt](http://example.com/image.jpg)" in result

    def test_extract_backgrounds(self, builder: ChatRunnerBuilder) -> None:
        """Test extracting background information"""
        metadata_list = [
            {
                "source_id": "doc1",
                "chunk": [
                    {"chunk_context": "Context 1"},
                    {"chunk_context": "Context 2"},
                ],
            },
            {
                "source_id": "doc2",
                "chunk": [{"chunk_context": "Context 3"}],
            },
        ]

        backgrounds = builder._extract_backgrounds(metadata_list)
        assert "Context 1" in backgrounds
        assert "Context 2" in backgrounds
        assert "Context 3" in backgrounds

    @pytest.mark.asyncio
    async def test_exec_query_knowledge(
        self, builder: ChatRunnerBuilder, span: Span
    ) -> None:
        """Test executing knowledge query"""
        params = KnowledgeQueryParams(
            repo_ids=["repo1"],
            doc_ids=["doc1"],
            top_k=3,
            score_threshold=0.3,
            rag_type="AIUI-RAG2",
        )

        mock_result: dict[str, Any] = {"data": {"results": []}}

        with patch(
            "agent.service.builder.chat_builder.KnowledgePluginFactory"
        ) as mock_factory:
            mock_plugin = MagicMock()
            mock_plugin.run = AsyncMock(return_value=mock_result)
            mock_factory.return_value.gen.return_value = mock_plugin

            result = await builder.exec_query_knowledge(params, span)

            assert result == mock_result


class TestKnowledgeQueryParams:
    """Test KnowledgeQueryParams dataclass"""

    def test_knowledge_query_params_creation(self) -> None:
        """Test creating KnowledgeQueryParams"""
        params = KnowledgeQueryParams(
            repo_ids=["repo1"],
            doc_ids=["doc1"],
            top_k=3,
            score_threshold=0.3,
            rag_type="AIUI-RAG2",
        )

        assert params.repo_ids == ["repo1"]
        assert params.doc_ids == ["doc1"]
        assert params.top_k == 3
        assert params.score_threshold == 0.3
        assert params.rag_type == "AIUI-RAG2"
