from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict


class AbstractDownloader(BaseModel):
    """Abstract base class for data downloaders using Pydantic BaseModel.
    
    This class provides a base interface for all downloader implementations,
    combining Pydantic's validation and serialization capabilities with
    abstract method definitions.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, *args, **kwargs):
        """
        Initialize the downloader class.
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        super().__init__(**kwargs)

    async def run_task(self, task: Any):
        """
        Run a single task.
        Args:
            task: The task to run.

        Returns:
            None
        """
        pass

    async def run(self, *args, **kwargs):
        """
        Run the downloader.
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            None
        """
        pass