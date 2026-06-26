from typing import List, Optional, Any, Type, Dict
from pydantic import BaseModel, Field, ConfigDict

from src.dynamic import dynamic_manager

class Strategy(BaseModel):
    """Strategy base class."""
    name: str = Field(description="The name of the strategy")
    description: str = Field(description="The description of the strategy")
    factor_names: List[str] = Field(description="The names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, **kwargs) -> Any:
        """Call the factor with the given arguments."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    
class StrategyConfig(BaseModel):
    """Configuration for a strategy."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the strategy")
    description: str = Field(description="The description of the strategy")
    factor_names: List[str] = Field(description="The names of the factors")
    version: str = Field(description="The version of the strategy")
    
    cls: Optional[Type[Strategy]] = Field(default=None, description="The class of the strategy")
    instance: Optional[Strategy] = Field(default=None, description="The instance of the strategy")
    config: dict = Field(default_factory=dict, description="The configuration of the strategy")
    code: str = Field(default="", description="The code of the strategy")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        
        result = {
            "name": self.name,
            "description": self.description,
            "factor_names": self.factor_names,
            "version": self.version,
            
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
        }
        
        return result
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Validate the model from a dictionary."""
        name = data.get("name")
        description = data.get("description")
        factor_names = data.get("factor_names")
        version = data.get("version")
        
        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code, 
                        class_name=class_name,
                        base_class=Strategy,
                        context="strategy"
                    )
                except Exception as e:
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None
            
        config = data.get("config")
        instance = data.get("instance", None)
        
        return cls( 
            name=name, 
            description=description, 
            factor_names=factor_names, 
            version=version, 
            cls=cls_, 
            instance=instance, 
            config=config, 
            code=code
        )