from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FileShellInputs(BaseModel):
    open: bool = Field(...)


class SandboxInputs(BaseModel):
    workpath: str = Field(...)
    files: Optional[FileShellInputs] = Field(default=None)
    shell: Optional[FileShellInputs] = Field(default=None)


class FollowUpSuggestionInputs(BaseModel):
    open: bool = Field(...)
    suggestion_prompt: Optional[str] = Field(default=None)


class PropertiesInputs(BaseModel):
    repos: list[str] = Field(...)
    docs: Optional[list[str]] = Field(default=None)
    top_k: int = Field(...)
    min_score: float = Field(...)


class KnowledgeInputs(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    type: Literal["AIUI-RAG2", "CBG-RAG"] = Field(...)
    properties: PropertiesInputs = Field(...)

    @model_validator(mode="after")
    def validate_docs_by_type(self) -> "KnowledgeInputs":
        """根据 type 字段验证 properties.docs 字段是否必传"""
        if self.type == "CBG-RAG" and self.properties.docs is None:
            raise ValueError("docs field is required when type is 'CBG-RAG'")
        return self


class RagInputs(BaseModel):
    call_mode: Literal["auto", "must"] = Field(...)
    query_rewrite: bool = Field(...)
    query_rewrite_prompt: str = Field(...)
    knowledges: Optional[list[KnowledgeInputs]] = Field(default=None)


class ToolInputs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(...)
    ori_name: str = Field(...)
    description: str = Field(...)
    tool_schema: dict = Field(..., alias="schema", serialization_alias="schema")


class CusMcpServerInputs(BaseModel):
    server_url: str = Field(...)
    tools: list[ToolInputs] = Field(...)


class LinkMcpServerInputs(BaseModel):
    server_id: str = Field(...)
    tools: list[ToolInputs] = Field(...)


class LinkToolInputs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    name: str = Field(...)
    ori_name: str = Field(...)
    description: str = Field(...)
    link_tool_schema: dict = Field(..., alias="schema", serialization_alias="schema")


class PluginInputs(BaseModel):
    link_tools: Optional[list[LinkToolInputs]] = Field(default=None)
    link_mcp_servers: Optional[list[LinkMcpServerInputs]] = Field(default=None)
    cus_mcp_servers: Optional[list[CusMcpServerInputs]] = Field(default=None)


class RandomInputs(BaseModel):
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[int] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)


class ParametersInputs(BaseModel):
    random: Optional[RandomInputs] = Field(default=None)


class ModelPropertiesInputs(BaseModel):
    id: str = Field(...)
    url: str = Field(...)
    token: str = Field(...)
    parameters: Optional[ParametersInputs] = Field(default=None)


class ModelInputs(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)
    type: str = Field(...)
    properties: ModelPropertiesInputs = Field(...)


class VariableInputs(BaseModel):
    key: str = Field(...)
    type: Literal["string", "object", "array"] = Field(...)
    description: str = Field(...)
    default_value: Union[str, dict, list] = Field(...)

    @model_validator(mode="after")
    def validate_default_value_type(self) -> "VariableInputs":
        """根据 type 字段验证 default_value 的类型"""
        expected_type = {"string": str, "object": dict, "array": list}[self.type]
        if not isinstance(self.default_value, expected_type):
            raise ValueError(
                f"default_value must be {expected_type.__name__} when type is '{self.type}', "
                f"got {type(self.default_value).__name__}"
            )
        return self


class PromptInputs(BaseModel):
    variables: Optional[list[VariableInputs]] = Field(default=None)
    prompt: str = Field(...)


class DslInputs(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)
    prompt: Optional[PromptInputs] = Field(default=None)
    max_reason_cnt: Optional[int] = Field(default=None)
    model: ModelInputs = Field(...)
    plugin: Optional[PluginInputs] = Field(default=None)
    rag: Optional[RagInputs] = Field(default=None)
    follow_up_suggestion: Optional[FollowUpSuggestionInputs] = Field(default=None)
    sandbox: Optional[SandboxInputs] = Field(default=None)


class ProtocolSynchronization(BaseModel):
    id: str = Field(...)
    dsl: Optional[DslInputs] = Field(default=None)


class Publish(BaseModel):
    bot_id: str = Field(...)
    version: str = Field(...)
    description: str = Field(...)
    dsl: Optional[DslInputs] = Field(default=None)


class Auth(BaseModel):
    version_id: int = Field(...)
    app_id: str = Field(...)
