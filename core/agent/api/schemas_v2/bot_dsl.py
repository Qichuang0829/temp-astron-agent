from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# prompt --
class PromptVariable(BaseModel):
    key: str = Field(...)
    type: Literal["string", "object", "array"] = Field(...)
    description: str = Field(...)
    default_value: Union[str, dict, list] = Field(...)

    @model_validator(mode="after")
    def validate_default_value_type(self) -> "PromptVariable":
        """Validate default_value type based on type field"""
        expected_type = {"string": str, "object": dict, "array": list}[self.type]
        if not isinstance(self.default_value, expected_type):
            raise ValueError(
                f"default_value must be {expected_type.__name__} when type is '{self.type}', "
                f"got {type(self.default_value).__name__}"
            )
        return self


class Prompt(BaseModel):
    variables: list[PromptVariable] = Field(default_factory=list)
    instruct: str = Field(default="")


# model --
class OpenAILikeModelPropertiesInputs(BaseModel):
    id: str = Field(...)
    url: str = Field(...)
    bearer_token: str = Field(...)


class Model(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    type: Literal["openai-like"] = Field(...)
    properties: Union[OpenAILikeModelPropertiesInputs] = Field(...)


# link --
class FcSchemaParameterProperty(BaseModel):
    name: str = Field(...)
    type: str = Field(...)
    description: str = Field(...)


class FcSchemaParameters(BaseModel):
    type: Literal["object"] = Field(...)
    properties: dict[str, FcSchemaParameterProperty] = Field(...)
    required: list[str] = Field(...)


class FcSchema(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    parameters: FcSchemaParameters = Field(...)


class LinkToolInputs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    version: str = Field(...)
    operation_id: str = Field(...)
    fc_schema: FcSchema = Field(...)


# mcp --
class McpToolInputs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    operation_id: str = Field(...)
    fc_schema: FcSchema = Field(...)


class LinkMcpServerInputs(BaseModel):
    server_id: str = Field(...)
    tools: list[McpToolInputs] = Field(...)


# custom mcp server --
class CusMcpServerInputs(BaseModel):
    server_url: str = Field(...)
    tools: list[McpToolInputs] = Field(...)


# workflow --
class WorkflowInputs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    flow_id: str = Field(...)
    fc_schema: FcSchema = Field(...)


class PluginInputs(BaseModel):
    link_tools: list[LinkToolInputs] = Field(default_factory=list)
    link_mcp_servers: list[LinkMcpServerInputs] = Field(default_factory=list)
    cus_mcp_servers: list[CusMcpServerInputs] = Field(default_factory=list)
    workflows: list[WorkflowInputs] = Field(default_factory=list)


# rag --
class RagKnowledgeProperties(BaseModel):
    repos: list[str] = Field(default_factory=list)
    docs: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3)
    min_score: float = Field(default=3.0)


class RagKnowledge(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    type: Literal["AIUI-RAG2", "CBG-RAG"] = Field(...)
    properties: RagKnowledgeProperties = Field(...)

    @model_validator(mode="after")
    def validate_docs_by_type(self) -> "RagKnowledge":
        """Validate whether properties.docs field is required based on type field"""
        if self.type == "CBG-RAG" and self.properties.docs is None:
            raise ValueError("docs field is required when type is 'CBG-RAG'")
        return self


class Rag(BaseModel):
    knowledges: list[RagKnowledge] = Field(default_factory=list)


# dsl --
class BotDsl(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    prompt: Prompt = Field(default_factory=Prompt)
    max_reason_cnt: int = Field(default=12)
    model: Model = Field(...)
    plugin: PluginInputs = Field(default_factory=PluginInputs)
    rag: Rag = Field(default_factory=Rag)
