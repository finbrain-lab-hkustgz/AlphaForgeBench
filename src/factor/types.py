from typing import Optional, Any,  List, Type, Dict
from pydantic import BaseModel, Field, ConfigDict


from src.dynamic import dynamic_manager

class Factor(BaseModel):
    """Base class for all factors."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(description="The type of the factor")
    expression: str = Field(description="The expression of the factor")
    description: str = Field(description="The description of the factor")
    names: List[str] = Field(description="The returned names of the factors")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, **kwargs) -> Any:
        """Call the factor with the given arguments."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    
class FactorConfig(BaseModel):
    """Configuration for a factor."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(description="The type of the factor")
    expression: str = Field(description="The expression of the factor")
    description: str = Field(description="The description of the factor")
    names: List[str] = Field(description="The returned names of the factors")
    version: str = Field(description="The version of the factor")
    
    cls: Optional[Type[Factor]] = Field(default=None, description="The class of the factor")
    instance: Optional[Factor] = Field(default=None, description="The instance of the factor")
    config: dict = Field(default_factory=dict, description="The configuration of the factor")
    code: str = Field(default="", description="The code of the factor")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        
        result = {
            "type": self.type,
            "expression": self.expression,
            "description": self.description,
            "names": self.names,
            "version": self.version,
            
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "config": self.config,
            "instance": None,
            "code": self.code,
        }
        
        return result
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'FactorConfig':
        """Validate the model from a dictionary."""
        type = data.get("type")
        expression = data.get("expression")
        description = data.get("description")
        names = data.get("names")
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
                        base_class=Factor,
                        context="factor"
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
            type=type, 
            expression=expression, 
            description=description, 
            names=names, 
            version=version, 
            cls=cls_,
            instance=instance,
            config=config,
            code=code
        )