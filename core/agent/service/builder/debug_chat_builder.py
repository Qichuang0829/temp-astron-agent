import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

from common.otlp.trace.span import Span

from agent.api.schemas_v2.bot_chat_inputs import DebugChat
from agent.service.builder.base_builder import BaseApiBuilder
from agent.service.runner.debug_chat_runner import DebugChatRunner
from agent.service.plugin.bot_plugins import BotPluginFactory
from agent.engine.bot.runner import BotRunner


@dataclass
class KnowledgeQueryParams:
    """Knowledge query parameters"""

    repo_ids: list[str]
    doc_ids: list[str]
    top_k: int
    score_threshold: float
    rag_type: str


class DebugChatRunnerBuilder(BaseApiBuilder):
    inputs: DebugChat

    async def build(self) -> DebugChatRunner:
        """Build"""
        with self.span.start("BuildRunner") as sp:
            model = await self.create_model(
                app_id=self.app_id,
                model_name=self.inputs.dsl.model.properties.id,
                base_url=self.inputs.dsl.model.properties.url,
                api_key=self.inputs.dsl.model.properties.bearer_token,
            )

            # build link plugins
            plugins = await BotPluginFactory(
                app_id=self.app_id,
                uid=self.uid,
                dsl=self.inputs.dsl).gen_tools(sp)

            return DebugChatRunner(
                bot_runner=BotRunner(
                    model=model,
                    chat_history=self.inputs.get_chat_history(),
                    plugins=plugins,
                    instruct=self.inputs.dsl.prompt.instruct,
                    question=self.inputs.get_last_message_content(),
                )
            )
