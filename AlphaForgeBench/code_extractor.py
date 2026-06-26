"""Extract strategy and factor code from LLM responses."""

import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

import dirtyjson

from src.logger import logger


# Known non-ASCII field name mappings (various languages -> English)
NON_ASCII_FIELD_MAPPINGS = {
    # Bengali
    "নাম": "name",
    "বর্ণনা": "description",
    # Telugu
    "పేరు": "name",
    "వివరణ": "description",
    # Malayalam
    "പേര്": "name",
    "വിവരണം": "description",
    # Armenian
    "անdelays": "name",
    "անdelays": "name",
    "նdelays": "description",
    # Hindi
    "नाम": "name",
    "विवरण": "description",
    # Tamil
    "பெயர்": "name",
    "விளக்கம்": "description",
    # Arabic
    "اسم": "name",
    "وصف": "description",
    # Russian
    "имя": "name",
    "описание": "description",
    # Chinese
    "名称": "name",
    "描述": "description",
    # Japanese
    "名前": "name",
    "説明": "description",
    # Korean
    "이름": "name",
    "설명": "description",
    # Greek
    "όνομα": "name",
    "περιγραφή": "description",
    # Thai
    "ชื่อ": "name",
    "คำอธิบาย": "description",
}


@dataclass
class CodeFix:
    """Record of a code fix applied."""

    query_id: str
    model: str
    sample_id: int
    fix_type: str  # e.g., "non_ascii_field_name"
    original_text: str
    fixed_text: str
    line_number: int | None = None
    details: dict = field(default_factory=dict)


@dataclass
class CodeFixReport:
    """Collection of code fixes for reporting."""

    fixes: List[CodeFix] = field(default_factory=list)

    def add_fix(self, fix: CodeFix):
        self.fixes.append(fix)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        fixes_by_model = {}
        for fix in self.fixes:
            model = fix.model
            if model not in fixes_by_model:
                fixes_by_model[model] = []
            fixes_by_model[model].append({
                "query_id": fix.query_id,
                "sample_id": fix.sample_id,
                "fix_type": fix.fix_type,
                "original_text": fix.original_text,
                "fixed_text": fix.fixed_text,
                "line_number": fix.line_number,
                "details": fix.details,
            })

        return {
            "total_fixes": len(self.fixes),
            "fixes_by_model": {
                model: {
                    "count": len(fixes),
                    "fixes": fixes,
                }
                for model, fixes in fixes_by_model.items()
            },
            "summary": {
                "by_fix_type": self._count_by_fix_type(),
                "by_model": {model: len(fixes) for model, fixes in fixes_by_model.items()},
            },
        }

    def _count_by_fix_type(self) -> dict:
        counts = {}
        for fix in self.fixes:
            counts[fix.fix_type] = counts.get(fix.fix_type, 0) + 1
        return counts

    def save(self, path: Path):
        """Save report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class CodeFixer:
    """Fix common issues in LLM-generated code."""

    # Pattern to match field definitions with non-ASCII names
    # Matches: `non_ascii_name: type = Field(...)`
    FIELD_PATTERN = re.compile(
        r'^(\s*)([^\x00-\x7F]+)\s*:\s*(.*?=\s*Field\(.*?\))',
        re.MULTILINE
    )

    # Pattern to match any identifier that contains non-ASCII characters
    NON_ASCII_IDENTIFIER = re.compile(r'[^\x00-\x7F]+')

    def __init__(self):
        self.report = CodeFixReport()

    def fix_code(
        self,
        code: str,
        query_id: str = "",
        model: str = "",
        sample_id: int = 0,
    ) -> Tuple[str, List[CodeFix]]:
        """
        Fix common issues in code.

        Args:
            code: The code to fix
            query_id: Query ID for reporting
            model: Model name for reporting
            sample_id: Sample ID for reporting

        Returns:
            Tuple of (fixed_code, list_of_fixes)
        """
        if not code:
            return code, []

        fixes = []
        fixed_code = code

        # Fix non-ASCII field names
        fixed_code, non_ascii_fixes = self._fix_non_ascii_field_names(
            fixed_code, query_id, model, sample_id
        )
        fixes.extend(non_ascii_fixes)

        # Add fixes to report
        for fix in fixes:
            self.report.add_fix(fix)

        return fixed_code, fixes

    def _fix_non_ascii_field_names(
        self,
        code: str,
        query_id: str,
        model: str,
        sample_id: int,
    ) -> Tuple[str, List[CodeFix]]:
        """Fix non-ASCII field names in Pydantic models."""
        fixes = []
        lines = code.split('\n')
        fixed_lines = []

        for line_num, line in enumerate(lines, 1):
            fixed_line = line

            # Check if line contains a field definition with non-ASCII name
            # Pattern: `non_ascii: type = Field(...)`
            match = re.match(r'^(\s*)([^\x00-\x7F]+)\s*:\s*(.+)$', line)
            if match:
                indent = match.group(1)
                non_ascii_name = match.group(2)
                rest = match.group(3)

                # Try to find the English equivalent
                english_name = self._get_english_field_name(non_ascii_name, rest)

                if english_name:
                    fixed_line = f"{indent}{english_name}: {rest}"
                    fixes.append(CodeFix(
                        query_id=query_id,
                        model=model,
                        sample_id=sample_id,
                        fix_type="non_ascii_field_name",
                        original_text=line.strip(),
                        fixed_text=fixed_line.strip(),
                        line_number=line_num,
                        details={
                            "original_name": non_ascii_name,
                            "fixed_name": english_name,
                        },
                    ))

            fixed_lines.append(fixed_line)

        return '\n'.join(fixed_lines), fixes

    def _get_english_field_name(self, non_ascii_name: str, field_definition: str) -> str | None:
        """
        Determine the English field name for a non-ASCII name.

        Uses:
        1. Known mappings
        2. Context from field definition (e.g., default value)
        """
        # Check known mappings
        if non_ascii_name in NON_ASCII_FIELD_MAPPINGS:
            return NON_ASCII_FIELD_MAPPINGS[non_ascii_name]

        # Try to infer from context
        field_def_lower = field_definition.lower()

        # Check for common field patterns
        if 'str' in field_def_lower and 'field(' in field_def_lower:
            # Check default value for hints
            if 'default=' in field_def_lower:
                # If it's a string field with a descriptive default, likely "name" or "description"
                if 'description' in field_def_lower or 'desc' in field_def_lower:
                    return "description"
                else:
                    return "name"

        if 'list[str]' in field_def_lower and 'factor' in field_def_lower:
            return "factor_names"

        # Default: if it looks like a name field (str type, first field), use "name"
        if 'str' in field_def_lower:
            return "name"

        return None

    def get_report(self) -> CodeFixReport:
        """Get the accumulated fix report."""
        return self.report

    def reset_report(self):
        """Reset the fix report."""
        self.report = CodeFixReport()


@dataclass
class ExtractedCode:
    """Extracted code from LLM response."""

    strategy_code: Optional[str] = None
    factor_codes: List[str] = None
    raw_response: str = ""
    extraction_method: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        if self.factor_codes is None:
            self.factor_codes = []

    @property
    def has_strategy(self) -> bool:
        return self.strategy_code is not None and len(self.strategy_code.strip()) > 0

    def to_dict(self) -> dict:
        return {
            "strategy_code": self.strategy_code,
            "factor_codes": self.factor_codes,
            "extraction_method": self.extraction_method,
            "error": self.error,
        }


class CodeExtractor:
    """Extract strategy and factor code from LLM responses."""

    # Forbidden imports that should be removed from generated code
    FORBIDDEN_IMPORTS = {
        'datasets',  # Hugging Face datasets
        'transformers',
        'torch',
        'tensorflow',
        'keras',
        'sklearn',
        'scipy',
        'matplotlib',
        'seaborn',
        'plotly',
    }

    # Patterns for extracting code
    JSON_PATTERN = re.compile(r'\{[^{}]*"strategy"[^{}]*\{[^{}]*"code"[^{}]*\}[^{}]*\}', re.DOTALL)
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\s*(.*?)```', re.DOTALL)
    CLASS_PATTERN = re.compile(r'(class\s+\w+\s*\([^)]*(?:Strategy|Factor)[^)]*\)\s*:.*?)(?=\nclass\s|\Z)', re.DOTALL)
    STRATEGY_CLASS_PATTERN = re.compile(r'(class\s+\w+\s*\([^)]*Strategy[^)]*\)\s*:.*?)(?=\nclass\s|\Z)', re.DOTALL)
    FACTOR_CLASS_PATTERN = re.compile(r'(class\s+\w+\s*\([^)]*Factor[^)]*\)\s*:.*?)(?=\nclass\s|\Z)', re.DOTALL)

    def extract(self, response: str) -> ExtractedCode:
        """
        Extract code from LLM response.

        Tries multiple extraction methods in order:
        1. JSON format: {"strategy": {"code": "..."}}
        2. Python code blocks with class definitions
        3. Raw class definitions in text

        Args:
            response: Raw LLM response text

        Returns:
            ExtractedCode object with extracted code
        """
        if not response or not response.strip():
            return ExtractedCode(
                raw_response=response,
                error="Empty response",
            )

        # Try JSON extraction first
        result = self._extract_from_json(response)
        if result.has_strategy:
            result.strategy_code = self._remove_forbidden_imports(result.strategy_code)
            return result

        # Try code block extraction
        result = self._extract_from_code_blocks(response)
        if result.has_strategy:
            result.strategy_code = self._remove_forbidden_imports(result.strategy_code)
            return result

        # Try raw class extraction
        result = self._extract_raw_classes(response)
        if result.has_strategy:
            result.strategy_code = self._remove_forbidden_imports(result.strategy_code)
            return result

        return ExtractedCode(
            raw_response=response,
            error="No strategy code found in response",
        )

    def _extract_from_json(self, response: str) -> ExtractedCode:
        """Extract code from JSON format response."""
        try:
            # Try to find JSON object in response
            # First, try to parse the entire response as JSON
            try:
                data = dirtyjson.loads(response)
                if isinstance(data, dict) and "strategy" in data:
                    strategy_code = data["strategy"].get("code", "")
                    factor_codes = []

                    # Extract factor codes if present
                    factors = data.get("factors", [])
                    for factor in factors:
                        if isinstance(factor, dict) and "code" in factor:
                            factor_codes.append(factor["code"])

                    if strategy_code:
                        return ExtractedCode(
                            strategy_code=strategy_code,
                            factor_codes=factor_codes,
                            raw_response=response,
                            extraction_method="json_full",
                        )
            except (json.JSONDecodeError, dirtyjson.Error):
                pass

            # Try to find JSON object after </think> tag
            think_end = response.find("</think>")
            if think_end != -1:
                json_part = response[think_end + 8:].strip()
                try:
                    data = dirtyjson.loads(json_part)
                    if isinstance(data, dict) and "strategy" in data:
                        strategy_code = data["strategy"].get("code", "")
                        if strategy_code:
                            return ExtractedCode(
                                strategy_code=strategy_code,
                                factor_codes=[],
                                raw_response=response,
                                extraction_method="json_after_think",
                            )
                except (json.JSONDecodeError, dirtyjson.Error):
                    pass

            # Try to find JSON object anywhere in response
            json_start = response.find('{"strategy"')
            if json_start == -1:
                json_start = response.find('{ "strategy"')

            if json_start != -1:
                # Find matching closing brace
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(response[json_start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = json_start + i + 1
                            break

                if json_end > json_start:
                    json_str = response[json_start:json_end]
                    try:
                        data = dirtyjson.loads(json_str)
                        if isinstance(data, dict) and "strategy" in data:
                            strategy_code = data["strategy"].get("code", "")
                            if strategy_code:
                                return ExtractedCode(
                                    strategy_code=strategy_code,
                                    factor_codes=[],
                                    raw_response=response,
                                    extraction_method="json_partial",
                                )
                    except (json.JSONDecodeError, dirtyjson.Error):
                        pass

        except Exception as e:
            logger.debug(f"JSON extraction failed: {e}")

        return ExtractedCode(raw_response=response, error="JSON extraction failed")

    def _extract_from_code_blocks(self, response: str) -> ExtractedCode:
        """Extract code from markdown code blocks."""
        code_blocks = self.CODE_BLOCK_PATTERN.findall(response)

        if not code_blocks:
            return ExtractedCode(raw_response=response, error="No code blocks found")

        strategy_code = None
        factor_codes = []

        for lang, block in code_blocks:
            block = block.strip()

            # Handle json code blocks or content starting with json prefix
            if lang == 'json' or block.startswith('json\n') or block.startswith('json\r\n'):
                # Remove possible json prefix (for backward compatibility)
                json_content = block
                if block.startswith('json\n'):
                    json_content = block[5:]
                elif block.startswith('json\r\n'):
                    json_content = block[6:]

                if '"strategy"' in json_content:
                    try:
                        data = dirtyjson.loads(json_content)
                        if isinstance(data, dict) and "strategy" in data:
                            code = data["strategy"].get("code", "")
                            if code:
                                return ExtractedCode(
                                    strategy_code=code,
                                    factor_codes=[],
                                    raw_response=response,
                                    extraction_method="json_code_block",
                                )
                    except (json.JSONDecodeError, dirtyjson.Error):
                        pass
                continue  # Skip further processing for json blocks

            # Check if this is a JSON code block containing strategy (no language identifier)
            if block.startswith('{') and '"strategy"' in block:
                try:
                    data = dirtyjson.loads(block)
                    if isinstance(data, dict) and "strategy" in data:
                        code = data["strategy"].get("code", "")
                        if code:
                            return ExtractedCode(
                                strategy_code=code,
                                factor_codes=[],
                                raw_response=response,
                                extraction_method="json_code_block",
                            )
                except (json.JSONDecodeError, dirtyjson.Error):
                    pass

            # Check for Strategy class
            strategy_matches = self.STRATEGY_CLASS_PATTERN.findall(block)
            if strategy_matches:
                # Take the full code block if it contains a Strategy class
                strategy_code = self._add_imports_if_needed(block)

            # Check for Factor classes
            factor_matches = self.FACTOR_CLASS_PATTERN.findall(block)
            for factor_match in factor_matches:
                factor_codes.append(self._add_imports_if_needed(factor_match))

        if strategy_code:
            return ExtractedCode(
                strategy_code=strategy_code,
                factor_codes=factor_codes,
                raw_response=response,
                extraction_method="code_block",
            )

        return ExtractedCode(raw_response=response, error="No Strategy class in code blocks")

    def _extract_raw_classes(self, response: str) -> ExtractedCode:
        """Extract class definitions directly from response text."""
        strategy_matches = self.STRATEGY_CLASS_PATTERN.findall(response)
        factor_matches = self.FACTOR_CLASS_PATTERN.findall(response)

        if not strategy_matches:
            return ExtractedCode(raw_response=response, error="No Strategy class found")

        # Take the first strategy class found
        strategy_code = self._add_imports_if_needed(strategy_matches[0].strip())
        factor_codes = [self._add_imports_if_needed(f.strip()) for f in factor_matches]

        return ExtractedCode(
            strategy_code=strategy_code,
            factor_codes=factor_codes,
            raw_response=response,
            extraction_method="raw_class",
        )

    def _add_imports_if_needed(self, code: str) -> str:
        """Add necessary imports if not present in code."""
        imports = []

        if "Strategy" in code and "from src.strategy.types import Strategy" not in code:
            imports.append("from src.strategy.types import Strategy")

        if "Factor" in code and "from src.factor.types import Factor" not in code:
            imports.append("from src.factor.types import Factor")

        if "Field" in code and "from pydantic import Field" not in code:
            imports.append("from pydantic import Field")

        if "pd." in code or "pd.DataFrame" in code:
            if "import pandas" not in code:
                imports.append("import pandas as pd")

        if "np." in code or "np.isnan" in code:
            if "import numpy" not in code:
                imports.append("import numpy as np")

        if imports:
            import_block = "\n".join(imports) + "\n\n"
            return import_block + code

        return code

    def _remove_forbidden_imports(self, code: str) -> str:
        """Remove forbidden import statements from code."""
        lines = code.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Check for "from X import ..." or "import X"
            should_remove = False
            for forbidden in self.FORBIDDEN_IMPORTS:
                if stripped.startswith(f'from {forbidden} import') or \
                   stripped.startswith(f'import {forbidden}') or \
                   stripped == f'import {forbidden}':
                    should_remove = True
                    break

            if not should_remove:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)


def extract_code_from_sample(sample_result: dict) -> ExtractedCode:
    """
    Extract code from a sample result dictionary.

    Args:
        sample_result: Dictionary with 'response' key

    Returns:
        ExtractedCode object
    """
    extractor = CodeExtractor()
    response = sample_result.get("response", "")
    return extractor.extract(response)
