from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict


class AbstractProcessor(BaseModel, ABC):
    """Abstract base class for data processors using Pydantic BaseModel.
    
    This class provides a base interface for all processor implementations,
    combining Pydantic's validation and serialization capabilities with
    abstract method definitions.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, *args, **kwargs):
        """
        Initialize the processor class.
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
        # Default implementation - can be overridden by subclasses that need it
        pass

    @abstractmethod
    async def run(self, *args, **kwargs):
        """
        Run the processor.
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            None
        """
        pass

