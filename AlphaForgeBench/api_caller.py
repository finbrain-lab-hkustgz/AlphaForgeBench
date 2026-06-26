"""LLM API caller for benchmark evaluation."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from src.model.manager import model_manager
from src.message.types import HumanMessage, SystemMessage
from src.logger import logger


@dataclass
class SampleResult:
    """Result of a single LLM sample."""

    query_id: str
    model: str
    sample_id: int
    temperature: float
    query_text: str
    response: str
    success: bool
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "model": self.model,
            "sample_id": self.sample_id,
            "temperature": self.temperature,
            "query_text": self.query_text,
            "response": self.response,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class LLMCaller:
    """Handles LLM API calls for benchmark evaluation."""

    def __init__(
        self,
        system_prompt: str,
        temperature: float = 0.8,
        max_concurrent: int = 5,
        retry_failed: bool = False,
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.retry_failed = retry_failed
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._initialized = False

    async def initialize(self):
        """Initialize the model manager."""
        if not self._initialized:
            await model_manager.initialize()
            self._initialized = True

    async def call_model(
        self,
        model: str,
        query: str,
        query_id: str,
        sample_id: int,
        max_retries: int = 3,
    ) -> SampleResult:
        """Call a single model with a query, with retry logic."""
        async with self._semaphore:
            last_error = None

            for attempt in range(max_retries):
                try:
                    messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=query),
                    ]

                    response = await model_manager(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                    )

                    if response.success and response.message:
                        return SampleResult(
                            query_id=query_id,
                            model=model,
                            sample_id=sample_id,
                            temperature=self.temperature,
                            query_text=query,
                            response=response.message,
                            success=True,
                        )
                    elif response.success and not response.message:
                        # API returned success but empty response, retry
                        last_error = "Empty response from API"
                        logger.warning(
                            f"Empty response for {model}/{query_id}, "
                            f"attempt {attempt + 1}/{max_retries}"
                        )
                    else:
                        # API returned error
                        last_error = response.message
                        logger.warning(
                            f"API error for {model}/{query_id}: {response.message}, "
                            f"attempt {attempt + 1}/{max_retries}"
                        )

                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        f"Exception for {model}/{query_id}: {e}, "
                        f"attempt {attempt + 1}/{max_retries}"
                    )

                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    import asyncio
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    await asyncio.sleep(wait_time)

            # All retries failed
            logger.error(f"All {max_retries} retries failed for {model}/{query_id}")
            return SampleResult(
                query_id=query_id,
                model=model,
                sample_id=sample_id,
                temperature=self.temperature,
                query_text=query,
                response="",
                success=False,
                error=f"Failed after {max_retries} retries: {last_error}",
            )

    async def generate_samples(
        self,
        model: str,
        queries: List[Dict[str, Any]],
        num_samples: int = 5,
        save_dir: Optional[Path] = None,
    ) -> List[SampleResult]:
        """
        Generate multiple samples for each query.

        Args:
            model: Model name to use
            queries: List of query dicts with level and query info
            num_samples: Number of samples per query (k in Pass@k)
            save_dir: Directory to save intermediate results (also used for cache lookup)

        Returns:
            List of SampleResult objects
        """
        await self.initialize()

        all_results = []
        tasks = []
        cached_count = 0

        for query_info in queries:
            query_id = query_info["query_id"]
            query_text = query_info.get("query", "")

            if not query_text:
                logger.warning(f"No query found for {query_id}")
                continue

            for sample_id in range(num_samples):
                # Check for existing sample (cache hit)
                if save_dir:
                    existing = self._load_existing_sample(
                        save_dir, model, query_id, sample_id
                    )
                    if existing:
                        all_results.append(existing)
                        cached_count += 1
                        continue

                task = self.call_model(
                    model=model,
                    query=query_text,
                    query_id=query_id,
                    sample_id=sample_id,
                )
                tasks.append(task)

        total_expected = len(queries) * num_samples

        if cached_count > 0:
            logger.info(f"  Loaded {cached_count}/{total_expected} samples from cache")

        if not tasks:
            logger.info(f"  All samples already cached, skipping API calls")
            return all_results

        # Execute all tasks concurrently with progress bar
        # Show progress including cached samples
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                f"[{model}]",
                total=total_expected,
                completed=cached_count,  # Start from cached count
            )

            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if isinstance(result, SampleResult):
                        all_results.append(result)
                        # Save intermediate result if requested
                        if save_dir:
                            self._save_sample(result, save_dir)
                except Exception as e:
                    logger.error(f"Task failed with exception: {e}")

                progress.advance(task_id)

        logger.info(f"  Generated {len(all_results) - cached_count} new samples")
        return all_results

    def _get_sample_filename(self, model: str, query_id: str, sample_id: int) -> str:
        """Generate filename for a sample result."""
        model_safe = model.replace("/", "_")
        return f"{model_safe}_{query_id}_{sample_id}.json"

    def _save_sample(self, result: SampleResult, save_dir: Path):
        """Save a single sample result to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = self._get_sample_filename(
            result.model, result.query_id, result.sample_id
        )
        filepath = save_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_existing_sample(
        self,
        save_dir: Path,
        model: str,
        query_id: str,
        sample_id: int,
    ) -> Optional[SampleResult]:
        """
        Check if a sample already exists and load it.

        Returns:
            SampleResult if exists (and valid), None otherwise
        """
        filename = self._get_sample_filename(model, query_id, sample_id)
        filepath = save_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # If retry_failed is enabled, check for failed samples:
            # 1. success=False (API error)
            # 2. success=True but response is empty (API returned empty content)
            if self.retry_failed:
                is_failed = not data.get("success", True)
                is_empty_response = data.get("success", False) and not data.get("response", "")
                if is_failed or is_empty_response:
                    reason = "API error" if is_failed else "empty response"
                    logger.info(f"Retrying failed sample ({reason}): {filename}")
                    return None

            return SampleResult(
                query_id=data["query_id"],
                model=data["model"],
                sample_id=data["sample_id"],
                temperature=data["temperature"],
                query_text=data["query_text"],
                response=data["response"],
                success=data["success"],
                error=data.get("error"),
                timestamp=data.get("timestamp", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to load existing sample {filepath}: {e}")
            return None


def load_queries_from_checkpoint(checkpoint_path: Path) -> List[Dict[str, Any]]:
    """
    Load queries from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint.json

    Returns:
        List of query dicts with query_id added
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for level, level_queries in data.items():
        if not isinstance(level_queries, list):
            continue

        for idx, query in enumerate(level_queries):
            query_with_id = query.copy()
            query_with_id["query_id"] = f"{level}_{idx}"
            query_with_id["level"] = level
            queries.append(query_with_id)

    return queries
