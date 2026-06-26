from typing import overload, Any, Union, List, Dict, Type
import base64
import os

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

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel

from src.message.types import (
    AssistantMessage,
    ContentPartAudio,
    ContentPartImage,
    ContentPartPdf,
    ContentPartRefusal,
    ContentPartText,
    ContentPartVideo,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool

from src.utils import assemble_project_path, encode_file_base64


class OpenRouterChatSerializer:
    """
    Serializer for converting between custom message types and OpenRouter chat completions API message param types.
    
    OpenRouter uses OpenAI-compatible API format, so this serializer is essentially the same as OpenAIChatSerializer.
    """

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
    def _serialize_content_part_audio(part: ContentPartAudio) -> dict[str, Any]:
        """Serialize audio content part for OpenRouter API.
        
        OpenRouter expects: {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}
        """
        audio_url = part.audio_url.url
        audio_format = part.audio_url.media_type.split('/')[-1]  # Extract format from media_type (e.g., "audio/mp3" -> "mp3")
        
        # Handle data URLs (base64 encoded)
        if audio_url.startswith("data:"):
            # Extract base64 data from data URL
            # Format: data:audio/mp3;base64,<base64_data>
            if "," in audio_url:
                header, data = audio_url.split(",", 1)
                # Extract format from header if available
                if "audio/" in header:
                    format_part = header.split("audio/")[1].split(";")[0]
                    if format_part:
                        audio_format = format_part
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": data,
                        "format": audio_format,
                    }
                }
        elif audio_url.startswith("file://"):
            # File path - read and encode to base64
            file_path = audio_url[7:]
            if not os.path.isabs(file_path):
                file_path = assemble_project_path(file_path)
            if os.path.exists(file_path):
                # Read file and encode to base64
                with open(file_path, "rb") as f:
                    audio_data = f.read()
                base64_data = base64.b64encode(audio_data).decode("utf-8")
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_data,
                        "format": audio_format,
                    }
                }
        elif os.path.exists(audio_url):
            # Direct file path
            with open(audio_url, "rb") as f:
                audio_data = f.read()
            base64_data = base64.b64encode(audio_data).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_data,
                    "format": audio_format,
                }
            }
        elif os.path.exists(assemble_project_path(audio_url)):
            # Relative file path
            file_path = assemble_project_path(audio_url)
            with open(file_path, "rb") as f:
                audio_data = f.read()
            base64_data = base64.b64encode(audio_data).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_data,
                    "format": audio_format,
                }
            }
        else:
            # URL - OpenRouter may support URLs, but for now we'll try to use URL directly
            # Note: OpenRouter may require base64 data, so this might need adjustment
            # For now, assume it's a URL that OpenRouter can handle
            return {
                "type": "input_audio",
                "input_audio": {
                    "url": audio_url,
                    "format": audio_format,
                }
            }

    @staticmethod
    def _serialize_content_part_video(part: ContentPartVideo) -> dict[str, Any]:
        """Serialize video content part for OpenRouter API.
        
        OpenRouter expects: {"type": "video_url", "video_url": {"url": "..."}}
        """
        return {
            "type": "video_url",
            "video_url": {
                "url": part.video_url.url,
            }
        }

    @staticmethod
    def _serialize_content_part_pdf(part: ContentPartPdf) -> dict[str, Any]:
        """Serialize PDF content part for OpenRouter API.
        
        OpenRouter expects: {"type": "file", "file": {"filename": "document.pdf", "file_data": data_url}}
        """
        pdf_url = part.pdf_url.url
        
        # Handle data URLs (base64 encoded)
        if pdf_url.startswith("data:"):
            # Already a data URL, extract filename from URL or use default
            filename = "document.pdf"
            # Try to extract filename from URL if it's a file:// URL converted to data URL
            if "file://" in pdf_url or "filename=" in pdf_url:
                # Could parse filename if available
                pass
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": pdf_url,
                }
            }
        elif pdf_url.startswith("file://"):
            # File path - read and encode to base64
            file_path = pdf_url[7:]
            if not os.path.isabs(file_path):
                file_path = assemble_project_path(file_path)
            if os.path.exists(file_path):
                # Read file and encode to base64
                with open(file_path, "rb") as f:
                    pdf_data = f.read()
                base64_data = base64.b64encode(pdf_data).decode("utf-8")
                filename = os.path.basename(file_path)
                data_url = f"data:application/pdf;base64,{base64_data}"
                return {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": data_url,
                    }
                }
        elif os.path.exists(pdf_url):
            # Direct file path
            with open(pdf_url, "rb") as f:
                pdf_data = f.read()
            base64_data = base64.b64encode(pdf_data).decode("utf-8")
            filename = os.path.basename(pdf_url)
            data_url = f"data:application/pdf;base64,{base64_data}"
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data_url,
                }
            }
        elif os.path.exists(assemble_project_path(pdf_url)):
            # Relative file path
            file_path = assemble_project_path(pdf_url)
            with open(file_path, "rb") as f:
                pdf_data = f.read()
            base64_data = base64.b64encode(pdf_data).decode("utf-8")
            filename = os.path.basename(file_path)
            data_url = f"data:application/pdf;base64,{base64_data}"
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data_url,
                }
            }
        else:
            # URL - try to use as data URL if it's already base64 encoded
            # Otherwise, assume it's a URL that needs to be downloaded
            return {
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": pdf_url if pdf_url.startswith("data:") else f"data:application/pdf;base64,{pdf_url}",
                }
            }

    @staticmethod
    def _serialize_user_content(
        content: str | list[ContentPartText | ContentPartImage | ContentPartAudio | ContentPartVideo | ContentPartPdf],
    ) -> str | list[Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, dict[str, Any]]]:
        """Serialize content for user messages (text, images, audio, video, and PDF allowed)."""
        if isinstance(content, str):
            return content
        serialized_parts: list[Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam, dict[str, Any]]] = []
        for part in content:
            if part.type == 'text':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_text(part))
            elif part.type == 'image_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_image(part))
            elif part.type == 'audio_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_audio(part))
            elif part.type == 'video_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_video(part))
            elif part.type == 'pdf_url':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_pdf(part))
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
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_text(part))
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
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_text(part))
            elif part.type == 'refusal':
                serialized_parts.append(OpenRouterChatSerializer._serialize_content_part_refusal(part))
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
    def serialize_message(message: HumanMessage) -> ChatCompletionUserMessageParam: ...

    @overload
    @staticmethod
    def serialize_message(message: SystemMessage) -> ChatCompletionSystemMessageParam: ...

    @overload
    @staticmethod
    def serialize_message(message: AssistantMessage) -> ChatCompletionAssistantMessageParam: ...

    @staticmethod
    def serialize_message(message: Message) -> ChatCompletionMessageParam:
        """Serialize a custom message to an OpenRouter message param."""
        if isinstance(message, HumanMessage):
            user_result: ChatCompletionUserMessageParam = {
                'role': 'user',
                'content': OpenRouterChatSerializer._serialize_user_content(message.content),
            }
            if message.name is not None:
                user_result['name'] = message.name
            return user_result

        elif isinstance(message, SystemMessage):
            system_result: ChatCompletionSystemMessageParam = {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': OpenRouterChatSerializer._serialize_system_content(message.content),
                        "cache_control": {
                            "type": "ephemeral"
                        }
                    }
                ],
            }
            if message.name is not None:
                system_result['name'] = message.name
            return system_result

        elif isinstance(message, AssistantMessage):
            # Handle content serialization
            content = None
            if message.content is not None:
                content = OpenRouterChatSerializer._serialize_assistant_content(message.content)
            assistant_result: ChatCompletionAssistantMessageParam = {'role': 'assistant'}
            # Only add content if it's not None
            if content is not None:
                assistant_result['content'] = content
            if message.name is not None:
                assistant_result['name'] = message.name
            if message.refusal is not None:
                assistant_result['refusal'] = message.refusal
            if message.tool_calls:
                assistant_result['tool_calls'] = [OpenRouterChatSerializer._serialize_tool_call(tc) for tc in message.tool_calls]
            return assistant_result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    @staticmethod
    def serialize_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
        return [OpenRouterChatSerializer.serialize_message(m) for m in messages]

    @staticmethod
    def serialize_tool(tool: "Tool") -> Dict[str, Any]:
        """Serialize a Tool instance to an OpenRouter tool param."""
        return tool.function_calling
    
    @staticmethod
    def serialize_tools(tools: List["Tool"]) -> List[Dict[str, Any]]:
        """Serialize a list of Tool instances to an OpenRouter tools param."""
        return [OpenRouterChatSerializer.serialize_tool(tool) for tool in tools]
    
    @staticmethod
    def serialize_response_format(
        response_format: Union[Type[BaseModel], BaseModel]
    ) -> Dict[str, Any]:
        """
        Format response_format from Pydantic model to OpenRouter-compatible JSON schema format.
        
        This function:
        1. Resolves $ref references
        2. Adds additionalProperties: false to all object types (OpenRouter requirement)
        3. Ensures all properties are in required array (OpenRouter strict mode requirement)
        
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
            2. Add additionalProperties: false to all object types (OpenRouter requirement)
            3. Preserve types, descriptions, and default values
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
                            # Resolve the reference and recursively optimize
                            return optimize_schema(defs[def_name], defs)
                
                # Process all keys
                for key, value in obj.items():
                    # Skip $ref as we handle it above
                    if key == "$ref":
                        continue
                    
                    # Recursively process nested structures
                    if key == "items":
                        # Process items and ensure it has type if it's a dict
                        processed_items = optimize_schema(value, defs)
                        if isinstance(processed_items, dict) and "type" not in processed_items:
                            # If items is a dict without type, infer it from context
                            # Check if it has properties (object) or other indicators
                            if "properties" in processed_items:
                                processed_items = {**processed_items, "type": "object"}
                            elif "$ref" in processed_items:
                                # Keep $ref as is, it will be resolved
                                pass
                            else:
                                # Default to "string" for simple types
                                processed_items = {**processed_items, "type": "string"}
                        optimized[key] = processed_items
                    elif key in ["properties"]:
                        optimized[key] = optimize_schema(value, defs)
                    elif key == "anyOf" or key == "oneOf" or key == "allOf":
                        # Handle union types
                        processed_items = [optimize_schema(item, defs) for item in value] if isinstance(value, list) else value
                        # Fix required fields in oneOf/anyOf items - ensure required only contains keys that exist in properties
                        if isinstance(processed_items, list):
                            for item in processed_items:
                                if isinstance(item, dict):
                                    properties = item.get("properties", {})
                                    required = item.get("required", [])
                                    if required and properties:
                                        # Filter required to only include keys that exist in properties
                                        item["required"] = [r for r in required if r in properties]
                        optimized[key] = processed_items
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
                
                # CRITICAL: Fix required fields - ensure required only contains keys that exist in properties
                if "required" in optimized and "properties" in optimized:
                    properties = optimized["properties"]
                    required = optimized["required"]
                    if isinstance(required, list) and isinstance(properties, dict):
                        # Filter required to only include keys that exist in properties
                        optimized["required"] = [r for r in required if r in properties]
                
                # CRITICAL: Add additionalProperties: false to ALL objects for OpenRouter
                if optimized.get("type") == "object":
                    optimized["additionalProperties"] = False
                # Also handle root level if it has properties but no explicit type
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
        
        # CRITICAL: Fix required fields at root level - ensure required only contains keys that exist in properties
        if "required" in optimized_schema and "properties" in optimized_schema:
            properties = optimized_schema["properties"]
            required = optimized_schema["required"]
            if isinstance(required, list) and isinstance(properties, dict):
                # Filter required to only include keys that exist in properties
                optimized_schema["required"] = [r for r in required if r in properties]
        
        # Ensure root schema has additionalProperties: false if it's an object
        if optimized_schema.get("type") == "object" and "additionalProperties" not in optimized_schema:
            optimized_schema["additionalProperties"] = False
        # Also handle root level if it has properties but no explicit type
        elif "properties" in optimized_schema and "type" not in optimized_schema:
            optimized_schema["type"] = "object"
            optimized_schema["additionalProperties"] = False
        
        # Fix required array: OpenRouter requires that if 'required' exists,
        # it must include ALL properties. This means all fields in properties must be in required.
        def fix_required_array(obj: Any, defs: Dict[str, Any] = None) -> Any:
            """Fix required arrays to include all properties for OpenRouter."""
            if defs is None:
                defs = defs_lookup
            
            if isinstance(obj, dict):
                fixed = {}
                
                # Handle $ref references
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]
                        if def_name in defs:
                            return fix_required_array(defs[def_name], defs)
                
                # Process all keys
                for key, value in obj.items():
                    if key == "$ref":
                        continue
                    elif isinstance(value, (dict, list)):
                        fixed[key] = fix_required_array(value, defs)
                    else:
                        fixed[key] = value
                
                # Fix required array for object types
                if fixed.get("type") == "object" and "properties" in fixed:
                    properties = fixed.get("properties", {})
                    
                    # OpenRouter requires ALL properties to be in required array
                    all_property_keys = list(properties.keys())
                    if all_property_keys:
                        fixed["required"] = all_property_keys
                    elif "required" in fixed:
                        # If no properties, keep empty required array
                        fixed["required"] = []
                
                return fixed
            elif isinstance(obj, list):
                return [fix_required_array(item, defs) for item in obj]
            else:
                return obj
        
        # Fix required arrays in the optimized schema
        optimized_schema = fix_required_array(optimized_schema)
        
        # Build the response format dictionary
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": optimized_schema,
            },
        }