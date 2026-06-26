from typing import overload, Any

try:
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartRefusalParam,
        ChatCompletionContentPartTextParam,
        ChatCompletionMessageFunctionToolCallParam,
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
    )
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL as OpenAIImageURL
    from openai.types.chat.chat_completion_message_function_tool_call_param import Function as OpenAIFunction
except ImportError:
    # Fallback types if openai package is not available
    ChatCompletionAssistantMessageParam = dict
    ChatCompletionContentPartImageParam = dict
    ChatCompletionContentPartRefusalParam = dict
    ChatCompletionContentPartTextParam = dict
    ChatCompletionMessageFunctionToolCallParam = dict
    ChatCompletionMessageParam = dict
    ChatCompletionSystemMessageParam = dict
    ChatCompletionUserMessageParam = dict
    OpenAIImageURL = dict
    OpenAIFunction = dict

from typing import Optional, List, Dict, Any, Union, Type, TYPE_CHECKING
from pydantic import BaseModel

from src.message.types import (
    AssistantMessage,
    ContentPartImage,
    ContentPartRefusal,
    ContentPartText,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
)

if TYPE_CHECKING:
    from src.tool.types import Tool


class OpenAIChatSerializer:
    """Serializer for converting between custom message types and OpenAI chat completions API message param types."""

    @staticmethod
    def _serialize_content_part_text(part: ContentPartText) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(text=part.text, type='text')

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImage) -> ChatCompletionContentPartImageParam:
        return ChatCompletionContentPartImageParam(
            image_url=OpenAIImageURL(url=part.image_url.url, detail=part.image_url.detail),
            type='image_url',
        )

    @staticmethod
    def _serialize_content_part_refusal(part: ContentPartRefusal) -> ChatCompletionContentPartRefusalParam:
        return ChatCompletionContentPartRefusalParam(refusal=part.refusal, type='refusal')

    @staticmethod
    def _serialize_user_content(
        content: str | list[ContentPartText | ContentPartImage],
    ) -> str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam]:
        """Serialize content for user messages (text and images allowed)."""
        if isinstance(content, str):
            return content
        serialized_parts: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenAIChatSerializer._serialize_content_part_text(part))
            elif part.type == 'image_url':
                serialized_parts.append(OpenAIChatSerializer._serialize_content_part_image(part))
        return serialized_parts

    @staticmethod
    def _serialize_system_content(
        content: str | list[ContentPartText],
    ) -> str | list[ChatCompletionContentPartTextParam]:
        """Serialize content for system messages (text only)."""
        if isinstance(content, str):
            return content
        serialized_parts: list[ChatCompletionContentPartTextParam] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenAIChatSerializer._serialize_content_part_text(part))
        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: str | list[ContentPartText | ContentPartRefusal] | None,
    ) -> str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam] | None:
        """Serialize content for assistant messages (text and refusal allowed)."""
        if content is None:
            return None
        if isinstance(content, str):
            return content
        serialized_parts: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenAIChatSerializer._serialize_content_part_text(part))
            elif part.type == 'refusal':
                serialized_parts.append(OpenAIChatSerializer._serialize_content_part_refusal(part))
        return serialized_parts

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
        return ChatCompletionMessageFunctionToolCallParam(
            id=tool_call.id,
            function=OpenAIFunction(name=tool_call.function.name, arguments=tool_call.function.arguments),
            type='function',
        )

    # region - Serialize overloads

    @overload
    @staticmethod
    def serialize(message: HumanMessage) -> ChatCompletionUserMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> ChatCompletionSystemMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> ChatCompletionAssistantMessageParam: ...

    @staticmethod
    def serialize(message: Message) -> ChatCompletionMessageParam:
        """Serialize a custom message to an OpenAI message param."""
        if isinstance(message, HumanMessage):
            user_result: ChatCompletionUserMessageParam = {
                'role': 'user',
                'content': OpenAIChatSerializer._serialize_user_content(message.content),
            }
            if message.name is not None:
                user_result['name'] = message.name
            return user_result

        elif isinstance(message, SystemMessage):
            system_result: ChatCompletionSystemMessageParam = {
                'role': 'system',
                'content': OpenAIChatSerializer._serialize_system_content(message.content),
            }
            if message.name is not None:
                system_result['name'] = message.name
            return system_result

        elif isinstance(message, AssistantMessage):
            # Handle content serialization
            content = None
            if message.content is not None:
                content = OpenAIChatSerializer._serialize_assistant_content(message.content)
            assistant_result: ChatCompletionAssistantMessageParam = {'role': 'assistant'}
            # Only add content if it's not None
            if content is not None:
                assistant_result['content'] = content
            if message.name is not None:
                assistant_result['name'] = message.name
            if message.refusal is not None:
                assistant_result['refusal'] = message.refusal
            if message.tool_calls:
                assistant_result['tool_calls'] = [OpenAIChatSerializer._serialize_tool_call(tc) for tc in message.tool_calls]
            return assistant_result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    @staticmethod
    def serialize_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
        return [OpenAIChatSerializer.serialize(m) for m in messages]

    @staticmethod
    def serialize_tools(tools: List["Tool"]) -> List[Dict[str, Any]]:
        """
        Serialize tools for OpenAI API calls. Convert Tool instances to function call format.
        
        Args:
            tools: List of Tool instances
            
        Returns:
            List of function call format dicts
        """
        return [tool.function_calling for tool in tools]
    
    @staticmethod
    def serialize_response_format(
        response_format: Union[Type[BaseModel], BaseModel]
    ) -> Dict[str, Any]:
        """
        Format response_format from Pydantic model to OpenAI-compatible JSON schema format.
        
        OpenAI requires additionalProperties: false for all object types (similar to OpenRouter).
        
        Args:
            response_format: BaseModel class or instance
            
        Returns:
            Dictionary containing response format configuration with:
            - type: "json_schema"
            - json_schema: Contains name, strict mode, and optimized schema
        """
        # Get the BaseModel class if it's an instance
        if isinstance(response_format, BaseModel) and not isinstance(response_format, type):
            model_class = type(response_format)
        else:
            model_class = response_format
        
        # Get JSON schema from Pydantic model
        schema = model_class.model_json_schema()
        
        # Build a lookup for $defs to resolve references
        defs_lookup = schema.get("$defs", {})
        
        def optimize_schema(obj: Any, defs: Dict[str, Any] = None) -> Any:
            """
            Recursively process schema to:
            1. Resolve $ref references
            2. Add additionalProperties: false to all object types (OpenAI requirement)
            """
            if defs is None:
                defs = defs_lookup
            
            if isinstance(obj, dict):
                optimized = {}
                
                # Handle $ref references
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]
                        if def_name in defs:
                            return optimize_schema(defs[def_name], defs)
                
                # Process all keys
                for key, value in obj.items():
                    if key == "$ref":
                        continue
                    elif key == "items":
                        # Process items and ensure it has type if it's a dict
                        processed_items = optimize_schema(value, defs)
                        if isinstance(processed_items, dict) and "type" not in processed_items:
                            # If items is a dict without type, infer it from context
                            if "properties" in processed_items:
                                processed_items = {**processed_items, "type": "object"}
                            elif "$ref" not in processed_items:
                                # Default to "string" for simple types (skip if has $ref)
                                processed_items = {**processed_items, "type": "string"}
                        optimized[key] = processed_items
                    elif key in ["properties"]:
                        optimized[key] = optimize_schema(value, defs)
                    elif key == "anyOf" or key == "oneOf" or key == "allOf":
                        optimized[key] = [optimize_schema(item, defs) for item in value] if isinstance(value, list) else value
                    elif isinstance(value, (dict, list)):
                        optimized[key] = optimize_schema(value, defs)
                    else:
                        optimized[key] = value
                
                # CRITICAL: Ensure array items have type key (double-check after processing)
                if optimized.get("type") == "array" and "items" in optimized:
                    items = optimized["items"]
                    if isinstance(items, dict) and "type" not in items and "$ref" not in items:
                        # If items is a dict without type and no $ref, add default type
                        if "properties" in items:
                            optimized["items"] = {**items, "type": "object"}
                        else:
                            optimized["items"] = {**items, "type": "string"}
                
                # CRITICAL: Add additionalProperties: false to ALL objects for OpenAI
                if optimized.get("type") == "object":
                    optimized["additionalProperties"] = False
                elif "properties" in optimized and "type" not in optimized:
                    optimized["type"] = "object"
                    optimized["additionalProperties"] = False
                
                return optimized
            elif isinstance(obj, list):
                return [optimize_schema(item, defs) for item in obj]
            else:
                return obj
        
        # Optimize the entire schema
        optimized_schema = optimize_schema(schema)
        
        # Ensure root schema has additionalProperties: false if it's an object
        if optimized_schema.get("type") == "object" and "additionalProperties" not in optimized_schema:
            optimized_schema["additionalProperties"] = False
        elif "properties" in optimized_schema and "type" not in optimized_schema:
            optimized_schema["type"] = "object"
            optimized_schema["additionalProperties"] = False
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": optimized_schema,
            },
        }


class OpenAIResponseSerializer:
    """Serializer for converting between custom message types and OpenAI responses API input format."""

    @staticmethod
    def _serialize_content_part_text(part: ContentPartText) -> dict[str, Any]:
        """Serialize text content part for responses API."""
        return {
            "type": "input_text",
            "text": part.text,
        }

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImage) -> dict[str, Any]:
        """Serialize image content part for responses API."""
        return {
            "type": "input_image",
            "image_url": part.image_url.url,
        }

    @staticmethod
    def _serialize_content(
        content: str | list[ContentPartText | ContentPartImage],
    ) -> list[dict[str, Any]]:
        """Serialize content for responses API."""
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]
        
        serialized_parts: list[dict[str, Any]] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenAIResponseSerializer._serialize_content_part_text(part))
            elif part.type == 'image_url':
                serialized_parts.append(OpenAIResponseSerializer._serialize_content_part_image(part))
        
        return serialized_parts

    @staticmethod
    def serialize(message: Message) -> dict[str, Any]:
        """Serialize a custom message to OpenAI responses API input format."""
        if isinstance(message, HumanMessage):
            result: dict[str, Any] = {
                "role": "user",
                "content": OpenAIResponseSerializer._serialize_content(message.content),
            }
            if message.name is not None:
                result["name"] = message.name
            return result

        elif isinstance(message, SystemMessage):
            # System messages are typically included in the first user message or handled separately
            # For responses API, we'll include them as system role
            result: dict[str, Any] = {
                "role": "system",
                "content": OpenAIResponseSerializer._serialize_content(message.content),
            }
            if message.name is not None:
                result["name"] = message.name
            return result

        elif isinstance(message, AssistantMessage):
            # Assistant messages are typically not in input, but we serialize them for completeness
            result: dict[str, Any] = {
                "role": "assistant",
            }
            if message.content is not None:
                result["content"] = OpenAIResponseSerializer._serialize_content(message.content)
            if message.name is not None:
                result["name"] = message.name
            return result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    @staticmethod
    def serialize_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Serialize a list of messages to OpenAI responses API input format."""
        return [OpenAIResponseSerializer.serialize(m) for m in messages]

