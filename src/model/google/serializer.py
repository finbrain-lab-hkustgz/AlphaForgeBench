from typing import overload, Any, List, Union, Optional, Type, TYPE_CHECKING
import base64
import os

from typing import Optional, List, Dict, Any, Union
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

from src.utils import assemble_project_path, decode_file_base64


class GoogleChatSerializer:
    """
    Serializer for converting between custom message types and Google Gemini API format.
    
    Google Gemini API format:
    - system_instruction: string (top-level field, not in messages)
    - contents: list of {"role": "user"|"model", "parts": [...]}
    - parts: list of {"text": "..."} or {"inline_data": {"mime_type": "...", "data": "..."}}
    - function_calls: {"name": "...", "args": {...}}
    - function_response: {"name": "...", "response": {...}}
    """

    @staticmethod
    def _serialize_content_part_text(part: ContentPartText) -> dict[str, Any]:
        return {"text": part.text}

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImage) -> dict[str, Any]:
        """Serialize image content part for Google Gemini API.
        
        Google Gemini expects: {"inline_data": {"mime_type": "...", "data": "..."}}
        """
        image_url = part.image_url.url
        
        # Handle data URLs (base64 encoded)
        if image_url.startswith("data:"):
            # Extract media type and base64 data from data URL
            # Format: data:image/jpeg;base64,<base64_data>
            header, data = image_url.split(",", 1)
            mime_type = "image/jpeg"  # default
            if "image/" in header:
                extracted_type = header.split("image/")[1].split(";")[0]
                mime_type = f"image/{extracted_type}"
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": data,
                }
            }
        elif image_url.startswith("file://"):
            # File path - read and encode to base64
            file_path = image_url[7:]
            if not os.path.isabs(file_path):
                file_path = assemble_project_path(file_path)
            if os.path.exists(file_path):
                # Read file and encode to base64
                with open(file_path, "rb") as f:
                    image_data = f.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
                # Guess mime type from file extension
                import mimetypes
                guessed_type, _ = mimetypes.guess_type(file_path)
                if not guessed_type or not guessed_type.startswith("image/"):
                    mime_type = "image/jpeg"  # default
                else:
                    mime_type = guessed_type
                return {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_data,
                    }
                }
        elif os.path.exists(image_url):
            # Direct file path
            with open(image_url, "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(image_url)
            if not guessed_type or not guessed_type.startswith("image/"):
                mime_type = "image/jpeg"  # default
            else:
                mime_type = guessed_type
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_data,
                }
            }
        elif os.path.exists(assemble_project_path(image_url)):
            # Relative file path
            file_path = assemble_project_path(image_url)
            with open(file_path, "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode("utf-8")
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(file_path)
            if not guessed_type or not guessed_type.startswith("image/"):
                mime_type = "image/jpeg"  # default
            else:
                mime_type = guessed_type
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_data,
                }
            }
        else:
            # URL - try to decode if it's base64, otherwise raise error
            # Google Gemini doesn't support direct URLs, only base64 inline_data
            raise ValueError(f"Google Gemini API only supports base64-encoded images or local files. Got: {image_url}")

    @staticmethod
    def _serialize_user_content(
        content: Union[str, List[Union[ContentPartText, ContentPartImage]]],
    ) -> List[dict[str, Any]]:
        """Serialize content for user messages (text and images allowed).
        
        Google Gemini requires parts to always be an array.
        """
        serialized_parts: List[dict[str, Any]] = []
        
        if isinstance(content, str):
            # Convert string to text part
            serialized_parts.append({"text": content})
        else:
            # Process content parts
            for part in content:
                if part.type == 'text':
                    serialized_parts.append(GoogleChatSerializer._serialize_content_part_text(part))
                elif part.type == 'image_url':
                    serialized_parts.append(GoogleChatSerializer._serialize_content_part_image(part))
        
        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: Optional[Union[str, List[ContentPartText]]],
    ) -> List[dict[str, Any]]:
        """Serialize content for assistant messages (text only, function_calls handled separately).
        
        Google Gemini requires parts to always be an array.
        """
        serialized_parts: List[dict[str, Any]] = []
        
        if content is None:
            return serialized_parts
        
        if isinstance(content, str):
            # Convert string to text part
            serialized_parts.append({"text": content})
        else:
            # Process content parts
            for part in content:
                if part.type == 'text':
                    serialized_parts.append(GoogleChatSerializer._serialize_content_part_text(part))
        
        return serialized_parts

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
        """Serialize tool call for Google Gemini API.
        
        Google Gemini expects: {"function_call": {"name": "...", "args": {...}}}
        """
        import json
        try:
            args_data = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
        except json.JSONDecodeError:
            args_data = {}
        
        return {
            "function_call": {
                "name": tool_call.function.name,
                "args": args_data,
            }
        }

    @overload
    @staticmethod
    def serialize(message: HumanMessage) -> dict[str, Any]: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> dict[str, Any]: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> dict[str, Any]: ...

    @staticmethod
    def serialize(message: Message) -> dict[str, Any]:
        """Serialize a custom message to a Google Gemini message format."""
        if isinstance(message, HumanMessage):
            parts = GoogleChatSerializer._serialize_user_content(message.content)
            result: dict[str, Any] = {
                'role': 'user',
                'parts': parts,
            }
            return result

        elif isinstance(message, SystemMessage):
            # System messages are handled separately (top-level system_instruction field)
            # Return content string to indicate it should be extracted
            content = message.content
            if isinstance(content, str):
                return {'role': 'system', 'content': content}
            elif isinstance(content, list):
                # Extract text from content parts
                text_parts = []
                for part in content:
                    if isinstance(part, ContentPartText):
                        text_parts.append(part.text)
                return {'role': 'system', 'content': ' '.join(text_parts)}
            else:
                return {'role': 'system', 'content': str(content)}

        elif isinstance(message, AssistantMessage):
            parts = GoogleChatSerializer._serialize_assistant_content(message.content)
            result: dict[str, Any] = {'role': 'model'}
            
            # Add function calls to parts array
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    parts.append(GoogleChatSerializer._serialize_tool_call(tool_call))
            
            # Parts is always an array (may be empty)
            result['parts'] = parts
            
            return result

        else:
            raise ValueError(f'Unknown message type: {type(message)}')

    @staticmethod
    def serialize_messages(messages: List[Message]) -> tuple[Optional[str], List[dict[str, Any]]]:
        """
        Serialize messages to Google Gemini format.
        
        Returns:
            Tuple of (system_instruction, contents_list)
            system_instruction: Optional string for system prompt (extracted from SystemMessage)
            contents_list: List of message dicts (excluding SystemMessage)
        """
        system_instruction = None
        gemini_contents: List[dict[str, Any]] = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                # Extract system instruction
                serialized = GoogleChatSerializer.serialize(message)
                if serialized.get('content'):
                    if system_instruction is None:
                        system_instruction = serialized['content']
                    else:
                        system_instruction += "\n" + serialized['content']
            else:
                # Serialize user/model messages
                gemini_contents.append(GoogleChatSerializer.serialize(message))
        
        return system_instruction, gemini_contents

    @staticmethod
    def serialize_tools(tools: List["Tool"]) -> List[Dict[str, Any]]:
        """
        Serialize tools for Google Gemini API calls. Convert Tool instances to Google Gemini tools format.
        
        Google Gemini tools format:
        [
            {
                "function_declarations": [
                    {
                        "name": "...",
                        "description": "...",
                        "parameters": {
                            "type": "object",
                            "properties": {...},
                            "required": [...]
                        }
                    }
                ]
            }
        ]
        
        Args:
            tools: List of Tool instances
            
        Returns:
            List containing a single dict with function_declarations array
        """
        # Lazy import to avoid circular dependency
        from src.tool.types import Tool
        
        function_declarations = []
        for tool in tools:
            if isinstance(tool, Tool):
                # Convert Tool instance to Google Gemini format
                function_call = tool.function_calling
                function_def = function_call.get("function", {})
                
                # Google Gemini uses "parameters" (same as OpenAI)
                gemini_function = {
                    "name": function_def.get("name", tool.name),
                    "description": function_def.get("description", tool.description),
                    "parameters": function_def.get("parameters", {}),
                }
                function_declarations.append(gemini_function)
        
        # Google Gemini expects tools as a list with a single object containing function_declarations
        if function_declarations:
            return [{"function_declarations": function_declarations}]
        else:
            return []
    
    @staticmethod
    def serialize_response_format(
        response_format: Union[Type[BaseModel], BaseModel, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format response_format to Google Gemini's response_schema parameter.
        
        Google Gemini uses response_schema:
        - Format: JSON Schema object
        - Requires response_mime_type: "application/json"
        
        Args:
            response_format: BaseModel class, instance, or dict
            
        Returns:
            Dictionary containing response_schema configuration:
            - response_schema: JSON schema
            - response_mime_type: "application/json"
        """
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Pydantic model class - get JSON schema
            schema = response_format.model_json_schema()
            # Ensure additionalProperties: false for structured output
            if schema.get("type") == "object" and "additionalProperties" not in schema:
                schema["additionalProperties"] = False
            elif "properties" in schema and "type" not in schema:
                schema["type"] = "object"
                schema["additionalProperties"] = False
            
            return {
                'response_schema': schema,
                'response_mime_type': 'application/json'
            }
        elif isinstance(response_format, BaseModel):
            # BaseModel instance - get the class
            model_class = type(response_format)
            schema = model_class.model_json_schema()
            # Ensure additionalProperties: false
            if schema.get("type") == "object" and "additionalProperties" not in schema:
                schema["additionalProperties"] = False
            elif "properties" in schema and "type" not in schema:
                schema["type"] = "object"
                schema["additionalProperties"] = False
            
            return {
                'response_schema': schema,
                'response_mime_type': 'application/json'
            }
        elif isinstance(response_format, dict):
            # Dict format - check if it's already in response_schema format
            if "response_schema" in response_format:
                # Already in response_schema format
                return {
                    'response_schema': response_format.get('response_schema'),
                    'response_mime_type': response_format.get('response_mime_type', 'application/json')
                }
            elif "type" in response_format and "json_schema" in response_format:
                # Chat completions format - convert to Google Gemini response_schema format
                json_schema_obj = response_format["json_schema"]
                schema = json_schema_obj.get("schema", {})
                return {
                    'response_schema': schema,
                    'response_mime_type': 'application/json'
                }
            else:
                # Assume it's a schema dict - wrap it
                return {
                    'response_schema': response_format,
                    'response_mime_type': 'application/json'
                }
        else:
            raise ValueError(f"Unsupported response_format type: {type(response_format)}")

