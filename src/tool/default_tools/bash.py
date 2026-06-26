"""Bash tool for executing shell commands."""
import asyncio
from re import T
from pydantic import Field
from typing import Dict, Any

from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL

_BASH_TOOL_DESCRIPTION = """Execute bash commands in the shell. 

IMPORTANT:
- Use this tool to run system commands, scripts, or any bash operations. 
- Be careful with commands that modify the system or require elevated privileges. 
- For file operations, ALWAYS use ABSOLUTE paths to avoid path-related issues. 
- Input should be a VALID bash command string.
"""

@TOOL.register_module(force=True)
class BashTool(Tool):
    """A tool for executing bash commands asynchronously."""
    name: str = "bash"
    description: str = _BASH_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    timeout: int = Field(description="Timeout in seconds for command execution", default=30)
    
    def __init__(self, **kwargs):
        """A tool for executing bash commands asynchronously."""
        super().__init__(**kwargs)
    
    async def __call__(self, command: str, **kwargs) -> ToolResponse:
        """Execute a bash command asynchronously.
        
        Args:
            command (str): The command to execute. If file path is necessary, it should be an absolute path.
        """
        try:
            # Sanitize the command
            if not command.strip():
                return ToolResponse(success=False, message="Error: Empty command provided")
            
            # Use shell=True to handle complex commands properly
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResponse(success=False, message=f"Error: Command timed out after {self.timeout} seconds")
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()
            
            # Prepare result
            result = []
            if stdout_str:
                result.append(f"STDOUT:\n{stdout_str}")
            if stderr_str:
                result.append(f"STDERR:\n{stderr_str}")
            
            exit_code = process.returncode
            if exit_code != 0:
                result.append(f"Exit code: {exit_code}")
            
            # Format the result
            message = "\n\n".join(result) if result else f"Command completed with exit code: {exit_code}"
            
            return ToolResponse(success=True, message=message)
            
        except Exception as e:
            return ToolResponse(success=False, message=f"Error executing command: {str(e)}")