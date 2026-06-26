"""File editor tool for editing file contents with line-based operations."""

import os
from typing import Optional, List, Dict, Any
from pydantic import Field
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL
from src.logger import logger

_FILE_EDITOR_DESCRIPTION = """File editor tool for editing file contents with multiple operations.

🎯 BEST FOR: Editing text files with line-based operations:
- Replace specific line ranges with new content
- Append content to the end of a file
- Perform multiple edits in a single call

📋 Parameters:
- file_path: Path to the file to edit (required)
- edits: List of edit operations (required), each operation is a dict with:
  - start_line: Starting line number (1-indexed, optional)
  - end_line: Ending line number (inclusive, optional)
  - content: New content to insert (required)
  
  If start_line and end_line are not provided, content is appended to the end.
  If only start_line is provided, content is inserted at that line.
  If both are provided, lines from start_line to end_line are replaced with content.

💡 Examples:
- Append to end:
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [{"content": "## New Section\\n\\nContent here."}]}}

- Replace lines 10-15:
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [{"start_line": 10, "end_line": 15, "content": "New content"}]}}

- Insert at line 5:
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [{"start_line": 5, "content": "Inserted line\\n"}]}}

- Multiple edits (applied from bottom to top to preserve line numbers):
  {"name": "edit", "args": {"file_path": "/path/to/report.md", "edits": [
    {"start_line": 20, "end_line": 25, "content": "Replace section 2"},
    {"start_line": 5, "end_line": 10, "content": "Replace section 1"}
  ]}}
"""


@TOOL.register_module(force=True)
class FileEditorTool(Tool):
    """A tool for editing file contents with line-based operations."""

    name: str = "edit"
    description: str = _FILE_EDITOR_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    def __init__(self, **kwargs):
        """Initialize the file editor tool."""
        super().__init__(**kwargs)

    async def __call__(
        self, 
        file_path: str,
        edits: List[Dict[str, Any]],
        **kwargs
    ) -> ToolResponse:
        """
        Edit file contents with multiple operations.

        Args:
            file_path (str): ABSOLUTE path to the file to edit.
            edits (List[Dict]): List of edit operations. Each dict can have:
                - start_line (int, optional): Starting line number (1-indexed)
                - end_line (int, optional): Ending line number (inclusive)
                - content (str, required): Content to insert/replace
        
        Returns:
            ToolResponse: ToolResponse with edit results or error message.
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
            
            # Validate edits
            if not edits or not isinstance(edits, list):
                return ToolResponse(
                    success=False,
                    message="Error: edits must be a non-empty list of edit operations."
                )
            
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            original_lines = len(lines)
            
            # Validate and normalize each edit
            normalized_edits = []
            for i, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    return ToolResponse(
                        success=False,
                        message=f"Error: Edit at index {i} must be a dict."
                    )
                
                if 'content' not in edit:
                    return ToolResponse(
                        success=False,
                        message=f"Error: Edit at index {i} must have 'content' field."
                    )
                
                content = edit['content']
                start_line = edit.get('start_line')
                end_line = edit.get('end_line')
                
                # Ensure content ends with newline if not empty
                if content and not content.endswith('\n'):
                    content += '\n'
                
                normalized_edits.append({
                    'start_line': start_line,
                    'end_line': end_line,
                    'content': content,
                    'original_index': i
                })
            
            # Sort edits by start_line in descending order (apply from bottom to top)
            # This preserves line numbers for earlier edits
            def get_sort_key(edit):
                if edit['start_line'] is None:
                    return float('inf')  # Appends go last
                return -edit['start_line']  # Negative for descending order
            
            sorted_edits = sorted(normalized_edits, key=get_sort_key)
            
            # Apply edits
            edit_results = []
            for edit in sorted_edits:
                start_line = edit['start_line']
                end_line = edit['end_line']
                content = edit['content']
                
                # Split content into lines
                content_lines = content.splitlines(keepends=True)
                if content and not content_lines:
                    content_lines = [content]
                
                if start_line is None and end_line is None:
                    # Append to end
                    lines.extend(content_lines)
                    edit_results.append({
                        'action': 'append',
                        'lines_added': len(content_lines),
                        'at_line': len(lines) - len(content_lines) + 1
                    })
                elif start_line is not None and end_line is None:
                    # Insert at start_line (push existing lines down)
                    insert_idx = max(0, min(start_line - 1, len(lines)))
                    for j, line in enumerate(content_lines):
                        lines.insert(insert_idx + j, line)
                    edit_results.append({
                        'action': 'insert',
                        'at_line': insert_idx + 1,
                        'lines_added': len(content_lines)
                    })
                else:
                    # Replace lines from start_line to end_line
                    start_idx = max(0, start_line - 1)
                    end_idx = min(end_line, len(lines)) if end_line else start_idx + 1
                    
                    # Boundary handling
                    if start_idx > len(lines):
                        start_idx = len(lines)
                    if end_idx > len(lines):
                        end_idx = len(lines)
                    if start_idx > end_idx:
                        end_idx = start_idx
                    
                    lines_removed = end_idx - start_idx
                    
                    # Remove old lines and insert new ones
                    del lines[start_idx:end_idx]
                    for j, line in enumerate(content_lines):
                        lines.insert(start_idx + j, line)
                    
                    edit_results.append({
                        'action': 'replace',
                        'start_line': start_idx + 1,
                        'end_line': end_idx,
                        'lines_removed': lines_removed,
                        'lines_added': len(content_lines)
                    })
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            new_lines = len(lines)
            
            # Build result message
            result_msg = f"File edited: {file_path}\n"
            result_msg += f"Original lines: {original_lines} → New lines: {new_lines}\n\n"
            result_msg += "Edits applied:\n"
            for i, result in enumerate(edit_results, 1):
                if result['action'] == 'append':
                    result_msg += f"  {i}. Appended {result['lines_added']} lines at line {result['at_line']}\n"
                elif result['action'] == 'insert':
                    result_msg += f"  {i}. Inserted {result['lines_added']} lines at line {result['at_line']}\n"
                elif result['action'] == 'replace':
                    result_msg += f"  {i}. Replaced lines {result['start_line']}-{result['end_line']} ({result['lines_removed']} lines) with {result['lines_added']} lines\n"
            
            message = result_msg
            
            logger.info(f"| ✏️ Edited file {file_path}: {len(edit_results)} operations")
            return ToolResponse(
                success=True,
                message=message,
                extra=ToolExtra(
                    file_path=file_path,
                    data={
                        "original_lines": original_lines,
                        "new_lines": new_lines,
                        "edits_applied": len(edit_results),
                        "edit_results": edit_results
                    }
                )
            )
                
        except UnicodeDecodeError:
            return ToolResponse(
                success=False,
                message=f"Error: Cannot edit file as text (binary file?): {file_path}"
            )
        except Exception as e:
            logger.error(f"| ❌ Error editing file: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error editing file: {str(e)}\n{traceback.format_exc()}"
            )

