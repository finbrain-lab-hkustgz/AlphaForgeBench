"""File reader tool for reading file contents with optional line range."""

import os
from typing import Optional, Dict, Any
from pydantic import Field

from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL
from src.logger import logger

_FILE_READER_DESCRIPTION = """File reader tool for reading file contents.

🎯 BEST FOR: Reading text files with optional line range:
- Read entire file content
- Read specific line range (start_line to end_line)
- Useful for reviewing reports, logs, code files, etc.

📋 Parameters:
- file_path: Path to the file to read (required)
- start_line: Starting line number (optional, 1-indexed)
- end_line: Ending line number (optional, inclusive)

💡 Examples:
- Read entire file: {"name": "read", "args": {"file_path": "/path/to/report.md"}}
- Read lines 1-400: {"name": "read", "args": {"file_path": "/path/to/report.md", "start_line": 1, "end_line": 400}}
- Read from line 400 to end: {"name": "read", "args": {"file_path": "/path/to/report.md", "start_line": 400}}
"""


@TOOL.register_module(force=True)
class FileReaderTool(Tool):
    """A tool for reading file contents with optional line range."""

    name: str = "read"
    description: str = _FILE_READER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    def __init__(self, **kwargs):
        """Initialize the file reader tool."""
        super().__init__(**kwargs)

    async def __call__(
        self, 
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs
    ) -> ToolResponse:
        """
        Read file contents with optional line range.

        Args:
            file_path (str): ABSOLUTE path to the file to read.
            start_line (Optional[int]): Starting line number.
            end_line (Optional[int]): Ending line number.
        
        Returns:
            ToolResponse: ToolResponse with file contents or error message.
        """
        try:
            # Validate file path
            if not file_path or not file_path.strip():
                return ToolResponse(
                    success=False,
                    message="Error: file_path is required."
                )
            
            file_path = file_path.strip()
            
            # Check if file exists
            if not os.path.exists(file_path):
                return ToolResponse(
                    success=False,
                    message=f"Error: File not found: {file_path}"
                )
            
            # Check if it's a file
            if not os.path.isfile(file_path):
                return ToolResponse(
                    success=False,
                    message=f"Error: Path is not a file: {file_path}"
                )
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Handle line range
            if start_line is not None or end_line is not None:
                # Convert to 0-indexed, with automatic boundary handling
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else total_lines
                
                # Automatic boundary handling
                if start_idx < 0:
                    start_idx = 0
                if start_idx > total_lines:
                    start_idx = total_lines  # Will result in empty content
                if end_idx > total_lines:
                    end_idx = total_lines  # Cap to file length
                if end_idx < 0:
                    end_idx = 0
                
                # Handle edge case: start > end (after boundary adjustment)
                if start_idx >= end_idx:
                    if start_line and start_line > total_lines:
                        return ToolResponse(
                            success=True,
                            message=f"File: {file_path}\nNote: start_line ({start_line}) exceeds file length ({total_lines} lines). No content to display.",
                            extra=ToolExtra(
                                file_path=file_path,
                                data={
                                    "content": "",
                                    "total_lines": total_lines
                                }
                            )
                        )
                    # Empty range
                    return ToolResponse(
                        success=True,
                        message=f"File: {file_path}\nNote: Line range {start_line}-{end_line} is empty or invalid. File has {total_lines} lines.",
                        extra=ToolExtra(
                            file_path=file_path,
                            data={
                                "content": "",
                                "total_lines": total_lines
                            }
                        )
                    )
                
                # Extract lines
                selected_lines = lines[start_idx:end_idx]
                content = ''.join(selected_lines)
                
                # Add line numbers
                numbered_content = ""
                for i, line in enumerate(selected_lines, start=start_idx + 1):
                    numbered_content += f"{i:6}|{line}"
                
                # Note if boundary was adjusted
                adjusted_note = ""
                if end_line and end_line > total_lines:
                    adjusted_note = f" (requested end_line {end_line} adjusted to {total_lines})"
                
                logger.info(f"| 📖 Read file {file_path} lines {start_idx + 1}-{end_idx}")
                return ToolResponse(
                    success=True,
                    message=f"File: {file_path}\nLines: {start_idx + 1}-{end_idx} (of {total_lines} total){adjusted_note}\n\n{numbered_content}",
                    extra=ToolExtra(
                        file_path=file_path,
                        data={
                            "content": content,
                            "start_line": start_idx + 1,
                            "end_line": end_idx,
                            "total_lines": total_lines
                        }
                    )
                )
            else:
                # Return full content
                content = ''.join(lines)
                
                # Add line numbers for display
                numbered_content = ""
                for i, line in enumerate(lines, start=1):
                    numbered_content += f"{i:6}|{line}"
                    
                message = f"File: {file_path}\nTotal lines: {total_lines}\n\n{numbered_content}"
                
                logger.info(f"| 📖 Read file {file_path} ({total_lines} lines)")
                return ToolResponse(
                    success=True,
                    message=message,
                    extra=ToolExtra(
                        file_path=file_path,
                        data={
                            "content": content,
                            "total_lines": total_lines
                        }
                    )
                )
                
        except UnicodeDecodeError:
            return ToolResponse(
                success=False,
                message=f"Error: Cannot read file as text (binary file?): {file_path}"
            )
        except Exception as e:
            logger.error(f"| ❌ Error reading file: {e}")
            return ToolResponse(
                success=False,
                message=f"Error reading file: {str(e)}"
            )

