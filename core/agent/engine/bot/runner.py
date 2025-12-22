import asyncio
import json
import time
from turtle import screensize
from pydantic import BaseModel, Field
from common.otlp.trace.span import Span
from common.otlp.log_trace.node_trace_log import NodeTraceLog as NodeTrace
from agent.api.schemas.agent_response import AgentResponse, CotStep, BotAgentResponse
from agent.domain.models.base import BaseLLMModel
from agent.engine.nodes.base import RunnerBase
from typing import Any, AsyncIterator, Union
from agent.engine.bot.prompt import BOT_TEMPLATE
from agent.service.plugin.base import BasePlugin, PluginResponse
from agent.service.plugin.link import LinkPlugin
from agent.service.plugin.mcp import McpPlugin
from agent.service.plugin.workflow import WorkflowPlugin
from agent.service.plugin.knowledge import KnowledgePlugin
from agent.api.schemas.llm_message import LLMMessage, LLMMessages
from common.otlp.log_trace.base import Usage
from common.otlp.log_trace.node_log import Data, NodeLog
from common.exceptions.base import BaseExc
from agent.exceptions.cot_exc import CotFormatIncorrectExc
from common.utils.snowfake import get_id
from agent.exceptions.bot_exc import ReasonStepPluginNotFoundExc


class BotCotStep(BaseModel):
    tool_call_list: list = Field(default_factory=list)
    empty: bool = Field(default=False)


class BotRunner(RunnerBase):
    instruct: str = Field(default="")
    question: str = Field(default="")
    plugins: list[
        Union[BasePlugin, McpPlugin, LinkPlugin, WorkflowPlugin, KnowledgePlugin]
    ]
    max_loop: int = Field(default=12)

    async def run(
        self, span: Span, node_trace: NodeTrace
    ) -> AsyncIterator[BotAgentResponse]:
        with span.start("RunBotAgent") as sp:
            prompt = await self.create_prompt()
            print("bot prompt ------")
            print(prompt)

            loop_count = 0

            while self.max_loop > loop_count:
                loop_count += 1
                loop_prompt = prompt + await self.create_scratchpad_prompt()

                msgs = LLMMessages(
                    messages=[LLMMessage(role="user", content=loop_prompt)]
                )

                cot_step = BotCotStep(empty=True)
                yield_answer = False
                async for agent_response in self.read_response(
                    msgs, True, sp, node_trace
                ):
                    if agent_response.typ == "reasoning_content":
                        print("reasoning_content: ", agent_response)
                        # yield agent_response
                    elif agent_response.typ == "content":
                        print("content: ", agent_response)
                        # yield agent_response
                        yield_answer = True
                    elif agent_response.typ == "cot_step":
                        cot_step = agent_response.content
                        break

                if yield_answer:
                    return
                if cot_step.empty:
                    # todo 降级到Function Call
                    raise CotFormatIncorrectExc(
                        m="Model returned content format is incorrect"
                    )

                await self.plugin_check(cot_step.tool_call_list)

                # send concurrence msg
                yield BotAgentResponse(typ="tool_call", content=cot_step.tool_call_list)

                try:
                    results = await asyncio.gather(
                        *[
                            self.handle_tool_call(tool_call)
                            for tool_call in cot_step.tool_call_list
                        ],
                        return_exceptions=True,
                    )
                    # 检查是否有异常
                    exceptions = [r for r in results if isinstance(r, Exception)]
                    if exceptions:
                        # 记录所有异常到span
                        for exc in exceptions:
                            sp.record_exception(exc)
                        # 抛出第一个异常
                        raise exceptions[0]
                except BaseExc:
                    # 重新抛出BaseExc类型的异常（保留原始异常信息）
                    raise
                except Exception as e:
                    # 捕获其他类型的异常，记录到span并重新抛出
                    # todo 走降级流程
                    sp.record_exception(e)
                    raise
                yield 1
                break

    async def handle_tool_call(self, tool_call: dict):
        tool_name = tool_call.get("name", "")
        tool_arguments = tool_call.get("arguments", {})
        plugin = tool_call.get("plugin", None)
        if not plugin:
            pass

        if plugin.typ == "link":
            await self.run_link_plugin(plugin, tool_call)
        elif plugin.typ == "mcp":
            pass
        elif plugin.typ == "workflow":
            pass
        elif plugin.typ == "knowledge":
            pass

    async def create_prompt(self) -> str:
        prompt = BOT_TEMPLATE.replace("{now}", self.cur_time())
        prompt = prompt.replace(
            "{tools}", "\n".join([tool.schema_template for tool in self.plugins])
        )
        prompt += await self.create_user_instruct()
        return prompt

    async def create_user_instruct(self) -> str:
        instruct = ""
        if self.instruct:
            instruct += f"用户指令：\n{self.instruct}\n"

        instruct += f"Previous chat history:\n{await self.create_history_prompt()}\nQuestion: {self.question}\n"
        return instruct

    async def create_scratchpad_prompt(self) -> str:
        return ""

    async def read_response(
        self, messages: LLMMessages, first_loop: bool, span: Span, node_trace: NodeTrace
    ):
        span = Span()

        with span.start("MakingStep") as sp:

            thinks = ""
            answers = ""

            step_content = ""
            final_answer = False

            # node赋值
            node_id = ""
            node_sid = span.sid
            node_node_id = span.sid
            node_type = "LLM"
            node_name = "ReadResponse"
            node_start_time = int(round(time.time() * 1000))
            node_running_status = True
            node_data_input = {
                "read_response_input": json.dumps(messages.list(), ensure_ascii=False)
            }
            node_data_output = {}
            node_data_config = {}
            node_data_usage = Usage()

            final_answer_stack = ""
            final_answer_stack_flag = False

            async for chunk in self.model.stream(messages.list(), True, sp):
                delta = chunk.choices[0].delta.dict()
                reasoning_content = delta.get("reasoning_content", "") or ""
                content: str = delta.get("content", "") or ""
                thinks += reasoning_content
                answers += content

                node_data_usage.completion_tokens = 0
                node_data_usage.prompt_tokens = 0
                node_data_usage.total_tokens = 0
                if chunk.usage and chunk.usage.model_dump().get("total_tokens") != 0:
                    node_data_usage.completion_tokens = chunk.usage.model_dump().get(
                        "completion_tokens"
                    )
                    node_data_usage.prompt_tokens = chunk.usage.model_dump().get(
                        "prompt_tokens"
                    )
                    node_data_usage.total_tokens = chunk.usage.model_dump().get(
                        "total_tokens"
                    )

                # ------
                if reasoning_content:  # 思考模型阶段
                    yield BotAgentResponse(
                        typ="reasoning_content",
                        content=reasoning_content,
                        model=self.model.name,
                    )
                    continue

                if final_answer:  # 回答阶段
                    if not final_answer_stack_flag:
                        if "<" in final_answer_stack:
                            temp_answer_content = content.split("<")[0]
                            yield BotAgentResponse(
                                typ="content",
                                content=temp_answer_content,
                                model=self.model.name,
                            )
                            final_answer_stack += content.lstrip(temp_answer_content)
                            final_answer_stack_flag = True
                        else:
                            yield BotAgentResponse(
                                typ="content", content=content, model=self.model.name
                            )
                    else:
                        final_answer_stack += content
                        if len(final_answer_stack) >= len("<|FinalAnswerEnd|>"):
                            if "<|FinalAnswerEnd|>" in final_answer_stack:
                                temp_answer_content = final_answer_stack.split(
                                    "<|FinalAnswerEnd|>"
                                )[0]
                                yield BotAgentResponse(
                                    typ="content",
                                    content=temp_answer_content,
                                    model=self.model.name,
                                )
                                break
                            else:
                                temp_answer_content = final_answer_stack
                                yield BotAgentResponse(
                                    typ="content",
                                    content=temp_answer_content,
                                    model=self.model.name,
                                )
                                final_answer_stack_flag = False
                                continue
                        else:
                            continue

                step_content += content
                if "<|FinalAnswerBegin|>" in step_content:
                    final_answer = True
                    right_content = step_content.split("<|FinalAnswerBegin|>")[1]
                    if right_content:
                        yield BotAgentResponse(
                            typ="content", content=right_content, model=self.model.name
                        )
                    continue

                if "<|ActionCallEnd|>" in step_content:
                    bot_cot_step = await self.parse_cot_step(step_content)
                    yield BotAgentResponse(
                        typ="cot_step", content=bot_cot_step, model=self.model.name
                    )
                    break

                # ------

            node_end_time = int(round(time.time() * 1000))
            data_llm_output = answers
            node_trace.trace.append(
                NodeLog(
                    id=node_id,
                    sid=node_sid,
                    node_id=node_node_id,
                    node_name=node_name,
                    node_type=node_type,
                    start_time=node_start_time,
                    end_time=node_end_time,
                    duration=node_end_time - node_start_time,
                    running_status=node_running_status,
                    # logs=[],
                    data=Data(
                        input=node_data_input if node_data_input else {},
                        output=node_data_output if node_data_output else {},
                        config=node_data_config if node_data_config else {},
                        usage=node_data_usage,
                    ),
                )
            )

            sp.add_info_events({"step-think": thinks})
            sp.add_info_events({"step-content": answers})

    async def parse_cot_step(self, step_content: str) -> BotCotStep:
        if not (
            "<|ActionCallBegin|>" in step_content
            and "<|ActionCallEnd|>" in step_content
        ):
            raise CotFormatIncorrectExc(m="Model returned content format is incorrect")
        action_content = step_content.split("<|ActionCallBegin|>")[1].split(
            "<|ActionCallEnd|>"
        )[0]
        try:
            tool_call_list = json.loads(action_content)
        except Exception as e:
            raise CotFormatIncorrectExc(m="Model returned content format is incorrect")

        for tool_call in tool_call_list:
            tool_call["id"] = get_id()
        return BotCotStep(tool_call_list=tool_call_list)

    async def get_plugin(self, tool_name: str):
        for plugin in self.plugins:
            if plugin.name == tool_name:
                return plugin
        return None

    async def plugin_check(self, tool_call_list: list):
        for tool_call in tool_call_list:
            tool_name = tool_call.get("name", "")
            plugin = await self.get_plugin(tool_name)
            if not plugin:
                raise ReasonStepPluginNotFoundExc(m=f"Plugin {tool_name} not found")
            tool_call["plugin"] = plugin

    async def run_link_plugin(
        self, plugin: LinkPlugin, tool_call: dict, sp: Span
    ) -> BotAgentResponse:
        plugin_response = await plugin.run(tool_call.get("arguments", {}), sp)
        tool_call["plugin_response"] = plugin_response

    async def run_mcp_plugin(
        self, plugin: McpPlugin, tool_call: dict, sp: Span
    ) -> BotAgentResponse:

        response = {"code": 0, "message": "success", "data": {}}
        reasoning_content = ""
        content = ""
        async for plugin_response in plugin.run():
            if plugin_response.code != 0:
                response["code"] = plugin_response.code
                response["message"] = plugin_response.chunk_data.get("message", "")
            else:
                pass
        return BotAgentResponse(typ="tool_call_result", content=response)
