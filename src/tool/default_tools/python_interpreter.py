"""Python interpreter tool for executing Python code."""

from typing import Optional, Dict, Any, List

from pydantic import ConfigDict, Field
from src.tool.default_tools.executor import (
    LocalPythonExecutor,
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
)
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.logger import logger
from src.registry import TOOL

_PYTHON_INTERPRETER_TOOL_DESCRIPTION = """Execute Python code and return the output.
Use this tool to run Python scripts, perform calculations, or execute any Python code.
The tool provides a safe execution environment with access to standard Python libraries.
"""

@TOOL.register_module(force=True)
class PythonInterpreterTool(Tool):
    """A tool that can execute Python code."""
    name: str = "python_interpreter"
    description: str = _PYTHON_INTERPRETER_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")

    authorized_imports: Optional[List[str]] = Field(default=None, description="The authorized imports for the tool")
    base_python_tools: Optional[Dict[str, Any]] = Field(default=None, description="The base python tools for the tool")
    python_evaluator: Optional["LocalPythonExecutor"] = Field(default=None, description="The python evaluator for the tool")
    
    def __init__(self, **kwargs):
        """A tool that can execute Python code."""
        super().__init__(**kwargs)

        if self.authorized_imports is None:
            self.authorized_imports = list(BASE_BUILTIN_MODULES)
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.authorized_imports))

        if self.base_python_tools is None:
            self.base_python_tools = dict(BASE_PYTHON_TOOLS)

        # Add additional common libraries if not already in authorized_imports
        additional_imports = [
            "subprocess",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "scipy",
            "sklearn",
            "json",
            "csv",
            "os",
            "matplotlib",
            "matplotlib.pyplot",
        ]
        # Merge with existing authorized_imports
        final_authorized_imports = list(set(self.authorized_imports) | set(additional_imports))
        
        if self.python_evaluator is None:
            self.python_evaluator = LocalPythonExecutor(
                additional_authorized_imports=final_authorized_imports,
            )
            self.python_evaluator.send_tools(self.base_python_tools)

    async def __call__(self, code: str, **kwargs) -> ToolResponse:
        """
        Execute the provided Python code.

        Args:
            code (str): Python code to execute.
        """
        try:
            self.python_evaluator.state = {}
            code_output = self.python_evaluator(code)
            output = f"Stdout:\n{code_output.logs}\nOutput: {str(code_output.output)}"
            
            message = output
            
            logger.info(f"| ✅ Executed Python code: {code}")
            return ToolResponse(success=True, message=message)

        except Exception as e:
            return ToolResponse(success=False, message=f"Error: {str(e)}")
