from typing import Any, Optional, Union, List, Dict, ClassVar
import httpx

try:
    from anthropic import AsyncAnthropic, APIError, APIConnectionError, RateLimitError
    try:
        from anthropic import transform_schema
    except ImportError:
        transform_schema = None
except ImportError:
    AsyncAnthropic = None
    APIError = Exception
    APIConnectionError = Exception
    RateLimitError = Exception
    transform_schema = None

from pydantic import BaseModel, Field, ConfigDict



import json
from src.logger import logger
from src.model.types import LLMResponse, LLMExtra
from src.message.types import Message, HumanMessage, SystemMessage, AssistantMessage
from src.model.anthropic.serializer import AnthropicChatSerializer
from src.utils import truncate_dict
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool.types import Tool

class ChatAnthropic(BaseModel):
    """
    A wrapper that provides a unified interface for Anthropic chat completions.
    
    This class handles Anthropic API-specific formatting and provides methods for chat completions
    with support for tools and streaming.
    
    Note: Only certain models support output_format (structured outputs):
    - claude-sonnet-4-5-20250929 and newer models support output_format
    - Older models like claude-3-7-sonnet-20250219 and claude-sonnet-4-20250514 do not support it
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Models that support output_format (structured outputs)
    OUTPUT_FORMAT_SUPPORTED_MODELS: ClassVar[List[str]] = [
        'claude-sonnet-4-5-20250929',
        'claude-opus-4-1-20250805',  # Opus 4.1
        # Add newer models here as they become available
    ]

    # Model configuration
    model: str

    # Model params
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 16384

    # Client initialization parameters
    api_key: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    reasoning: Optional[Dict[str, Any]] = None
    timeout: Optional[Union[float, httpx.Timeout]] = None
    max_retries: int = 5
    default_headers: Optional[Dict[str, str]] = None
    http_client: Optional[httpx.AsyncClient] = None

    @property
    def provider(self) -> str:
        return 'anthropic'

    def _get_client_params(self) -> Dict[str, Any]:
        """Prepare client parameters dictionary."""
        # Prepare default headers
        headers = dict(self.default_headers) if self.default_headers else {}
        
        # Add Anthropic beta header for structured outputs support
        if 'anthropic-beta' not in headers:
            headers['anthropic-beta'] = 'structured-outputs-2025-11-13'
        
        base_params = {
            'api_key': self.api_key,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'default_headers': headers if headers else None,
        }
        
        # Add base_url if provided
        if self.base_url:
            base_params['base_url'] = str(self.base_url)
        
        # Add http_client if provided
        if self.http_client is not None:
            base_params['http_client'] = self.http_client
        
        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}
        
        return client_params

    def get_client(self) -> AsyncAnthropic:
        """
        Returns an AsyncAnthropic client.

        Returns:
            AsyncAnthropic: An instance of the AsyncAnthropic client.
        """
        if AsyncAnthropic is None:
            raise ImportError("anthropic package is required. Install it with: pip install anthropic")
        
        client_params = self._get_client_params()
        return AsyncAnthropic(**client_params)

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_usage(self, response) -> Optional[Dict[str, Any]]:
        """Extract usage information from Anthropic response."""
        if hasattr(response, 'usage') and response.usage is not None:
            return response.usage.model_dump()
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
            - system: System message (if any)
            - messages: Serialized messages
            - params: All other API parameters (tools, temperature, max_tokens, etc.)
        """
        # Serialize messages to Anthropic format
        system_message, anthropic_messages = AnthropicChatSerializer.serialize_messages(messages)
        
        # Build API parameters
        params: Dict[str, Any] = {
            'model': self.model,
            'messages': anthropic_messages,
        }
        
        # Add system message if provided
        if system_message:
            params['system'] = system_message
        
        # Add model parameters
        if self.temperature is not None:
            params['temperature'] = self.temperature
        if self.top_p is not None:
            params['top_p'] = self.top_p
        if self.max_tokens is not None:
            params['max_tokens'] = self.max_tokens
        if self.reasoning is not None:
            params.update(self.reasoning)
        
        # Format tools using serializer
        if tools:
            formatted_tools = AnthropicChatSerializer.serialize_tools(tools)
            if formatted_tools:
                params['tools'] = formatted_tools
        
        # Handle response_format (Anthropic uses output_format parameter with beta API)
        # Only certain models support output_format
        use_beta_api = False
        if response_format:
            # Check if model supports output_format
            model_supports_output_format = any(
                supported_model in self.model 
                for supported_model in ChatAnthropic.OUTPUT_FORMAT_SUPPORTED_MODELS
            )
            
            if not model_supports_output_format:
                logger.warning(
                    f"Model {self.model} does not support output_format. "
                    f"Supported models: {', '.join(ChatAnthropic.OUTPUT_FORMAT_SUPPORTED_MODELS)}. "
                    f"Skipping structured output."
                )
            else:
                try:
                    params['output_format'] = AnthropicChatSerializer.serialize_response_format(response_format)
                    use_beta_api = True
                except ValueError as e:
                    logger.warning(f"Failed to serialize response_format: {e}")
        
        # Add betas parameter if using structured outputs
        if use_beta_api:
            params['betas'] = ['structured-outputs-2025-11-13']
        
        # Handle streaming
        if stream:
            params['stream'] = True
            logger.warning("Streaming is not yet fully implemented for Anthropic API")
        
        # Merge additional kwargs
        params.update(kwargs)
        
        return {
            "system": system_message,
            "messages": anthropic_messages,
            "params": params,
            "use_beta_api": use_beta_api,
        }

    async def _call_model(
        self,
        use_beta_api: bool = False,
        **params: Any,
    ) -> Any:
        """
        Call the model API (Step 2).
        
        Unified interface for calling the Anthropic API.
        
        Args:
            use_beta_api: Whether to use beta API (for structured outputs)
            **params: API parameters (should include model, messages, system, temperature, etc.)
            
        Returns:
            Response object from Anthropic API
        """
        client = self.get_client()
        # Use beta API if output_format is provided
        if use_beta_api:
            response = await client.beta.messages.create(**params)
        else:
            response = await client.messages.create(**params)
        
        return response

    async def __call__(
        self,
        messages: List[Message],
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Execute asynchronous completion call via Anthropic API.

        Args:
            messages: List of Message objects (HumanMessage, SystemMessage, AssistantMessage)
            tools: Optional list of Tool instances
            response_format: Optional response format (Pydantic model class, instance or dict)
            stream: Whether to stream the response (not implemented yet)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with formatted message
        """
        if AsyncAnthropic is None:
            raise ImportError("anthropic package is required. Install it with: pip install anthropic")

        try:
            params = await self._build_params(
                messages=messages,
                tools=tools,
                response_format=response_format,
                stream=stream,
                **kwargs,
            )
            
            response = await self._call_model(
                use_beta_api=params.get("use_beta_api", False),
                **params["params"],
            )
            
            return await self._format_response(
                response=response,
                tools=tools,
                response_format=response_format,
            )

        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            return LLMResponse(
                success=False,
                message=f"Rate limit error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return LLMResponse(
                success=False,
                message=f"API connection error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )
        except APIError as e:
            logger.error(f"API error: {e}")
            return LLMResponse(
                success=False,
                message=f"API error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "status_code": getattr(e, 'status_code', None), "model": self.name})
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return LLMResponse(
                success=False,
                message=f"Unexpected error: {str(e)}",
                extra=LLMExtra(data={"error": str(e), "model": self.name})
            )

    async def _format_response(
        self,
        response: Any,
        tools: Optional[List["Tool"]] = None,
        response_format: Optional[Union[Type[BaseModel], BaseModel, Dict]] = None,
    ) -> LLMResponse:
        """Format Anthropic response into LLMResponse."""
        try:
            # Handle SDK response object
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, dict):
                content = response.get("content", [])
            else:
                content = []

            if not content:
                return LLMResponse(
                    success=False,
                    message="No content in response",
                    extra=LLMExtra(data={"raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)})
                )

            # Extract text content and tool calls
            text_parts = []
            tool_calls = []
            
            for item in content:
                if hasattr(item, 'type'):
                    # SDK response object
                    if item.type == "text":
                        text_parts.append(item.text)
                    elif item.type == "tool_use":
                        tool_calls.append({
                            "id": item.id,
                            "name": item.name,
                            "input": item.input,
                        })
                elif isinstance(item, dict):
                    # Dict format
                    item_type = item.get("type")
                    if item_type == "text":
                        text_parts.append(item.get("text", ""))
                    elif item_type == "tool_use":
                        tool_calls.append(item)

            message_text = "\n".join(text_parts) if text_parts else ""

            usage = self._get_usage(response)
            stop_reason = response.stop_reason if hasattr(response, 'stop_reason') else response.get("stop_reason") if isinstance(response, dict) else None

            # Handle function calling
            if tools and tool_calls:
                formatted_lines = []
                functions = []

                for tool_call in tool_calls:
                    name = tool_call.get("name", "")
                    tool_id = tool_call.get("id", "")
                    input_data = tool_call.get("input", {})

                    # Format arguments as keyword arguments
                    if input_data:
                        args_str = ", ".join([f"{k}={v!r}" for k, v in input_data.items()])
                        formatted_lines.append(f"Calling function {name}({args_str})")
                    else:
                        formatted_lines.append(f"Calling function {name}()")

                    functions.append({
                        "id": tool_id,
                        "name": name,
                        "args": input_data
                    })

                formatted_message = "\n".join(formatted_lines)

                extra = LLMExtra(
                    data={
                        "raw_response": response.model_dump() if hasattr(response, 'model_dump') else response,
                        "functions": functions,
                        "usage": usage,
                        "stop_reason": stop_reason,
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
                        extra=LLMExtra(data={"raw_response": response})
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
                            "raw_response": response.model_dump() if hasattr(response, 'model_dump') else response,
                            "usage": usage,
                            "stop_reason": stop_reason,
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
                        "raw_response": response.model_dump() if hasattr(response, 'model_dump') else response,
                        "usage": usage,
                        "stop_reason": stop_reason,
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

