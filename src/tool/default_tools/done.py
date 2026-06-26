"""Done tool for indicating that the task has been completed."""
from typing import Dict, Any
from pydantic import Field
from src.tool.types import Tool, ToolResponse
from src.registry import TOOL

_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed.
Use this tool to signal that a task or subtask has been finished.
Provide the result of the task in the result parameter.
"""

@TOOL.register_module(force=True)
class DoneTool(Tool):
    """A tool for indicating that the task has been completed."""

    name: str = "done"
    description: str = _DONE_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    def __init__(self, **kwargs):
        """A tool for indicating that the task has been completed."""
        super().__init__(**kwargs)

    async def __call__(self, result: str, **kwargs) -> ToolResponse:
        """
        Indicate that the task has been completed.

        Args:
            result (str): Summary of the accomplished task.
        """
        
        message = f"✅ Task completed with result: {result}"
        
        return ToolResponse(success=True, message=message)