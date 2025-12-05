"""
Example module demonstrating project structure and coding standards.

This module shows how to structure Python code following the project
guidelines defined in CLAUDE.md.
"""

import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


class ExampleClass:
    """Example class demonstrating proper documentation.
    
    This class serves as a template for writing well-documented,
    type-hinted Python code.
    
    Attributes:
        name: A string representing the instance name
        value: An integer value associated with the instance
    """
    
    def __init__(self, name: str, value: int = 0):
        """Initialize the ExampleClass.
        
        Args:
            name: The name for this instance
            value: Initial value (default: 0)
        """
        self.name = name
        self.value = value
    
    def process_data(self, data: list[int]) -> Optional[float]:
        """Process a list of integers and return the average.
        
        Args:
            data: List of integers to process
            
        Returns:
            Average of the input data, or None if empty
            
        Raises:
            ValueError: If data contains non-numeric values
        """
        if not data:
            logger.warning("Empty data list provided")
            return None
        
        try:
            result = sum(data) / len(data)
            logger.info(f"Processed {len(data)} items, average: {result}")
            return result
        except TypeError as e:
            logger.error(f"Invalid data type in list: {e}")
            raise ValueError("All data items must be numeric") from e


def example_function(x: int, y: int) -> int:
    """Add two integers and return the result.
    
    Args:
        x: First integer
        y: Second integer
        
    Returns:
        Sum of x and y
    """
    return x + y
