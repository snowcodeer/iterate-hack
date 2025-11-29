"""Base interface for game controllers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class GameController(ABC):
    """Abstract base class for game controllers."""
    
    @abstractmethod
    def start_game(self) -> bool:
        """Initialize and start the game.
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def apply_action(self, action: Any) -> Dict:
        """Apply an action to the game.
        
        Args:
            action: Action from RL agent
            
        Returns:
            Dict with action result info
        """
        pass
    
    @abstractmethod
    def get_game_state(self) -> Dict[str, float]:
        """Get current game state as normalized values.
        
        Returns:
            Dict of state variables (0-1 normalized)
        """
        pass
    
    @abstractmethod
    def reset_game(self) -> bool:
        """Reset game to initial state.
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def close_game(self):
        """Clean up and close game."""
        pass
    
    @abstractmethod
    def is_game_running(self) -> bool:
        """Check if game is still running.
        
        Returns:
            True if game is running
        """
        pass

