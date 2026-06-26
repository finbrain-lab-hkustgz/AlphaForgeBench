from typing import Any, Optional, Union, List, Dict, ClassVar
import httpx

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.api_core import exceptions as google_exceptions
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    google_exceptions = None

from pydantic import BaseModel, Field, ConfigDict

import json
from src.logger import logger
from src.model.types import LLMResponse, LLMExtra
from src.message.types import Message, HumanMessage, SystemMessage, AssistantMessage
from src.model.google.serializer import GoogleChatSerializer
from src.utils import truncate_dict
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool

class ChatGoogle(BaseModel):
    """
    A wrapper that provides a unified interface for Google Gemini chat completions.
    
    This class handles Google Gemini API-specific formatting and provides methods for chat completions
    with support for tools and streaming.
    
    Note: Google Gemini uses response_schema for structured outputs.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Model configuration
    model: str

    # Model params
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = 8192
    reasoning: Optional[Dict[str, Any]] = None
    
    # Client initialization parameters
    api_key: Optional[str] = None
    timeout: Optional[Union[float, httpx.Timeout]] = None
    max_retries: int = 5

    @property
    def provider(self) -> str:
        return 'google'

    def _get_client_params(self) -> Dict[str, Any]:
        """Prepare client parameters dictionary."""
        if genai is None:
            raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")
        
        # Configure API key
        if self.api_key:
            genai.configure(api_key=self.api_key)
        elif not genai.api_key:
            # Try to get from environment
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        return {}

    def get_client(self, system_instruction: Optional[str] = None):
        """
        Returns a Google GenerativeModel instance.

        Args:
            system_instruction: Optional system instruction to configure the model
        
        Returns:
            GenerativeModel: An instance of the GenerativeModel client.
        """
        if genai is None:
            raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")
        
        self._get_client_params()
        
        # system_instruction should be passed when creating the model, not in generate_content
        if system_instruction:
            return genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction
            )
        else:
            return genai.GenerativeModel(self.model)

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_usage(self, response) -> Optional[Dict[str, Any]]:
        """Extract usage information from Google Gemini response."""
        if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
            return response.usage_metadata.model_dump()
        else:
            return None

    async def _build_params(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build parameters for API call.
        
        Step 1: Convert messages, tools, and response_format into API-ready parameters.
        
        Args:
            messages: List of Message objects
            tools: Optional list of Tool instances
            response_format: Optional response format (Pydantic model class, instance or dict)
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing:
            - system_instruction: System instruction (if any)
            - contents: Serialized messages
            - generation_config: Generation configuration (temperature, max_output_tokens, etc.)
            - tools: Serialized tools (if any)
            - params: All other API parameters
        """
        # Serialize messages to Google Gemini format
        system_instruction, gemini_contents = GoogleChatSerializer.serialize_messages(messages)
        
        # Build generation config
        generation_config: Dict[str, Any] = {}
        
        if self.temperature is not None:
            generation_config['temperature'] = self.temperature
        if self.top_p is not None:
            generation_config['top_p'] = self.top_p
        if self.top_k is not None:
            generation_config['top_k'] = self.top_k
        if self.max_output_tokens is not None:
            generation_config['max_output_tokens'] = self.max_output_tokens
        if self.reasoning is not None:
            generation_config.update(self.reasoning)
        # Handle response_format (Google Gemini uses response_schema)
        if response_format:
            try:
                response_format_config = GoogleChatSerializer.serialize_response_format(response_format)
                generation_config.update(response_format_config)
            except ValueError as e:
                logger.warning(f"Failed to serialize response_format: {e}")
        
        # Format tools using serializer
        tools_config = None
        if tools:
            formatted_tools = GoogleChatSerializer.serialize_tools(tools)
            if formatted_tools:
                tools_config = formatted_tools
        
        # Merge additional kwargs into generation_config
        for key, value in kwargs.items():
            if key not in ['contents', 'system_instruction', 'tools', 'generation_config']:
                generation_config[key] = value
        
        return {
            "system_instruction": system_instruction,
            "contents": gemini_contents,
            "generation_config": generation_config,
            "tools": tools_config,
            "stream": stream,
        }

    async def _call_model(
        self,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Call the model API (Step 2).
        
        Unified interface for calling the Google Gemini API.
        
        Args:
            contents: List of message dicts
            system_instruction: Optional system instruction string (passed to model constructor)
            generation_config: Generation configuration dict
            tools: Optional tools configuration
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Response object from Google Gemini API
        """
        # system_instruction must be passed when creating the model, not in generate_content
        client = self.get_client(system_instruction=system_instruction)
        
        # Prepare parameters for generate_content
        call_kwargs: Dict[str, Any] = {}
        
        if contents:
            call_kwargs['contents'] = contents
        if generation_config:
            call_kwargs['generation_config'] = generation_config
        if tools:
            call_kwargs['tools'] = tools
        
        # Handle streaming
        if stream:
            # For streaming, use generate_content with stream=True
            # Note: Google Gemini streaming API might be different
            logger.warning("Streaming is not yet fully implemented for Google Gemini API")
            call_kwargs['stream'] = True
        
        # Call the API
        # Google Gemini uses generate_content for single requests
        # For chat, we can use start_chat() or generate_content() with contents
        response = await self._async_generate_content(client, **call_kwargs)
        
        return response

    async def _async_generate_content(self, client, **kwargs):
        """Async wrapper for Google Gemini generate_content."""
        import asyncio
        
        # Google Gemini SDK is synchronous, so we run it in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: client.generate_content(**kwargs))

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous completion call via Google Gemini API.

        Args:
            messages: List of Message objects (HumanMessage, SystemMessage, AssistantMessage)
            tools: Optional list of Tool instances
            response_format: Optional response format (Pydantic model class, instance or dict)
            stream: Whether to stream the response (not implemented yet)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with formatted message
        """
        if genai is None:
            raise ImportError("google-generativeai package is required. Install it with: pip install google-generativeai")

        try:
            params = await self._build_params(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )
            
            response = await self._call_model(
                contents=params["contents"],
                system_instruction=params.get("system_instruction"),
                generation_config=params.get("generation_config"),
                tools=params.get("tools"),
                stream=params.get("stream", False),
            )
            
            return await self._format_response(
                response=response,
                tools=tools,
                response_format=response_format,
            )

        except Exception as e:
            error_msg = str(e)
            status_code = None
            
            # Try to extract status code from Google API exceptions
            if google_exceptions and isinstance(e, google_exceptions.GoogleAPIError):
                status_code = getattr(e, 'status_code', None)
            
            logger.error(f"API error: {e}")
            return LLMResponse(
                success=False,
                message=f"API error: {error_msg}",
                extra=LLMExtra(data={"error": error_msg, "status_code": status_code, "model": self.name})
            )

    async def _format_response(
        self,
        response: Any,
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """Format Google Gemini response into LLMResponse."""
        try:
            # Handle SDK response object
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
            elif isinstance(response, dict):
                candidates = response.get("candidates", [])
                candidate = candidates[0] if candidates else {}
            else:
                candidate = {}

            if not candidate:
                return LLMResponse(
                    success=False,
                    message="No candidates in response",
                    extra=LLMExtra(data={"raw_response": str(response)})
                )

            # Extract content and function calls
            text_parts = []
            function_calls = []
            
            if hasattr(candidate, 'content'):
                # SDK response object
                content = candidate.content
                if hasattr(content, 'parts'):
                    parts = content.parts
                else:
                    parts = []
            elif isinstance(candidate, dict):
                # Dict format
                content = candidate.get("content", {})
                parts = content.get("parts", [])
            else:
                parts = []
            
            for part in parts:
                if hasattr(part, 'text'):
                    # SDK response object
                    text_parts.append(part.text)
                elif hasattr(part, 'function_call'):
                    # SDK response object - function call
                    func_call = part.function_call
                    function_calls.append({
                        "name": func_call.name if hasattr(func_call, 'name') else "",
                        "args": func_call.args if hasattr(func_call, 'args') else {},
                    })
                elif isinstance(part, dict):
                    # Dict format
                    if "text" in part:
                        text_parts.append(part.get("text", ""))
                    elif "function_call" in part:
                        func_call = part["function_call"]
                        function_calls.append({
                            "name": func_call.get("name", ""),
                            "args": func_call.get("args", {}),
                        })

            message_text = "\n".join(text_parts) if text_parts else ""

            usage = self._get_usage(response)
            finish_reason = None
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
            elif isinstance(candidate, dict):
                finish_reason = candidate.get("finish_reason")

            # Handle function calling
            if tools and function_calls:
                formatted_lines = []
                functions = []

                for func_call in function_calls:
                    name = func_call.get("name", "")
                    args_data = func_call.get("args", {})

                    # Format arguments as keyword arguments
                    if args_data:
                        args_str = ", ".join([f"{k}={v!r}" for k, v in args_data.items()])
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")

                    functions.append({
                        "name": name,
                        "args": args_data
                    })

                formatted_message = "\n".join(formatted_lines)

                extra = LLMExtra(
                    data={
                        "raw_response": str(response),
                        "functions": functions,
                        "usage": usage,
                        "finish_reason": finish_reason,
                    }
                )

                return LLMResponse(
                    success=True,
                    message=formatted_message,
                    extra=extra
                )

            # Handle structured output (if response_format was provided)
            elif response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
                if not message_text:
                    return LLMResponse(
                        success=False,
                        message="Empty response content from model",
                        extra=LLMExtra(data={"raw_response": str(response)})
                    )

                # Try to parse JSON from message text
                import json
                try:
                    data = json.loads(message_text)
                    parsed_model = response_format.model_validate(data)

                    # Format as string
                    model_name = response_format.__name__
                    model_dict = parsed_model.model_dump()

                    field_lines = []
                    for field_name, field_value in model_dict.items():
                        field_lines.append(f"{field_name}={field_value!r}")

                    formatted_message = f"Response result:\n\n{model_name}(\n"
                    formatted_message += ",\n".join(f"    {line}" for line in field_lines)
                    formatted_message += "\n)"

                    extra = LLMExtra(
                        parsed_model=parsed_model,
                        data={
                            "raw_response": str(response),
                            "usage": usage,
                            "finish_reason": finish_reason,
                        }
                    )

                    return LLMResponse(
                        success=True,
                        message=formatted_message,
                        extra=extra
                    )
                except json.JSONDecodeError as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to parse JSON from response: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": message_text})
                    )
                except Exception as e:
                    return LLMResponse(
                        success=False,
                        message=f"Failed to validate response against schema: {e}",
                        extra=LLMExtra(data={"error": str(e), "content": message_text})
                    )

            # Default: return content as string
            else:
                extra = LLMExtra(
                    data={
                        "raw_response": str(response),
                        "usage": usage,
                        "finish_reason": finish_reason,
                    }
                )

                return LLMResponse(
                    success=True,
                    message=message_text,
                    extra=extra
                )

        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return LLMResponse(
                success=False,
                message=f"Failed to format response: {e}",
                extra=LLMExtra(data={"error": str(e)})
            )

