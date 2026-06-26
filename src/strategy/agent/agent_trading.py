from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
import pandas as pd

from src.strategy.types import Strategy
from src.model import model_manager
from src.message.types import HumanMessage, SystemMessage
from src.logger import logger
from src.utils import dedent

class TradingDecision(BaseModel):
    """Trading decision model for structured output."""
    action: str = Field(description="The trading action: BUY, SELL, or HOLD")
    reasoning: str = Field(description="The reasoning behind the decision")
    
    
SYSTEM_PROMPT = dedent("""
    You are an expert quantitative trader. Analyze the provided financial data and decide the next action.
    
    Your reasoning process should follow these steps:
    1. **Data Analysis**: Thoroughly analyze the historical price data and technical indicators provided in the 'Price' section. Identify trends, support/resistance levels, and any significant patterns.
    2. **Execution History Review**: Examine the 'History Valid Action' section. Evaluate the effectiveness of past decisions and how the market responded to them.
    3. **Inventory & Constraints Check**: Check your current 'cash' and 'position'. 
       - **Strict Constraint**: If your cash is 0, you **cannot** BUY.
       - **Strict Constraint**: If your position is 0, you **cannot** SELL.
    4. **Final Decision**: Based on the above analysis, decide whether to BUY, SELL, or HOLD. Provide your reasoning clearly.
    """)

class AgentTradingStrategy(Strategy):
    """Agent-based trading strategy.
    
    This strategy uses an LLM agent to decide the trading action based on 
    a prompt provided in the state.
    """
    
    name: str = Field(default="agent_trading_strategy", description="The name of the strategy")
    description: str = Field(default="Trading strategy that uses an LLM agent to make decisions", description="The description of the strategy")
    factor_names: List[str] = Field(default=[], description="The names of the factors")
    
    model_name: str = Field(default="openrouter/gemini-3-flash-preview", description="LLM model name to use")
    system_prompt: str = Field(default=SYSTEM_PROMPT, description="System prompt for the agent")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def __call__(self, state: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """Generate trading signal using an LLM agent with structured output.
        
        Args:
            state (Union[Dict[str, Any], pd.DataFrame]): The state dictionary from AgentTradingEnvironment,
                containing:
                - prompt: The formatted markdown prompt for the agent
                - symbol: The asset symbol
                - asset_info: Asset metadata
        
        Returns:
            Dict[str, Any]: The dictionary containing the strategy outputs:
                - action: The trading action (-1, 0, 1)
                - signal: The trading signal (-1, 0, 1) for compatibility
                - position: Quantity of the order (default 1.0)
                - reason: The agent's reasoning for the action
        """
        if state is None:
            logger.warning("| ⚠️ AgentTradingStrategy: Received None state. Defaulting to HOLD.")
            return {"action": 0, "signal": 0, "position": 0.0, "reason": "Received None state"}

        if isinstance(state, pd.DataFrame):
            logger.warning("| ⚠️ AgentTradingStrategy: Received DataFrame but expected state dict. Defaulting to HOLD.")
            return {"action": 0, "signal": 0, "position": 0.0, "reason": "Expected state dict, got DataFrame"}

        prompt = state.get("prompt", "")
        if not prompt:
            logger.warning("| ⚠️ AgentTradingStrategy: No prompt found in state.")
            return {"action": 0, "signal": 0, "position": 0.0, "reason": "No prompt in state"}

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        logger.info(f"| 🤖 AgentTradingStrategy calling LLM ({self.model_name}) with structured output...")
        response = await model_manager(
            self.model_name, 
            messages=messages,
            response_format=TradingDecision
        )

        if not response.success:
            logger.error(f"| ❌ AgentTradingStrategy: LLM call failed: {response.message}")
            return {"action": 0, "signal": 0, "position": 0.0, "reason": f"LLM call failed: {response.message}"}

        # Use parsed_model if available
        decision = getattr(response.extra, 'parsed_model', None)
        if decision and isinstance(decision, TradingDecision):
            action_str = decision.action.upper()
            reasoning = decision.reasoning
        else:
            # Fallback if parsing didn't happen or extra is missing
            logger.warning(f"| ⚠️ AgentTradingStrategy: No parsed_model in response. Falling back to response.message.")
            # If model_manager didn't parse it, response.message might contain the raw text
            # But with response_format, it should have parsed it.
            return {"action": 0, "signal": 0, "position": 0.0, "reason": "Failed to get structured output"}
        
        signal = 0
        if "BUY" in action_str:
            signal = 1
        elif "SELL" in action_str:
            signal = -1
        else:
            signal = 0

        order = {
            "action": signal,
            "signal": signal,
            "position": 1.0,
            "reason": reasoning
        }
        
        logger.info(f"| ✅ AgentTradingStrategy decided: {action_str} (action: {signal})")
        return order

