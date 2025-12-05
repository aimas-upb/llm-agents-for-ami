"""
Tests for the example module.

Demonstrates testing best practices including:
- Clear test names
- Arrange-Act-Assert pattern
- Edge case testing
- Error handling verification
"""

import pytest
from src.example import ExampleClass, example_function


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    def test_initialization(self):
        """Test that ExampleClass initializes correctly."""
        obj = ExampleClass("test", 42)
        assert obj.name == "test"
        assert obj.value == 42
    
    def test_process_data_with_valid_input(self):
        """Test process_data with valid integer list."""
        obj = ExampleClass("test")
        result = obj.process_data([1, 2, 3, 4, 5])
        assert result == 3.0
    
    def test_process_data_with_empty_list(self):
        """Test process_data with empty list returns None."""
        obj = ExampleClass("test")
        result = obj.process_data([])
        assert result is None
    
    def test_process_data_with_invalid_data(self):
        """Test process_data raises ValueError for invalid data."""
        obj = ExampleClass("test")
        with pytest.raises(ValueError):
            obj.process_data([1, 2, "invalid", 4])


def test_example_function():
    """Test the example_function."""
    assert example_function(2, 3) == 5
    assert example_function(-1, 1) == 0
    assert example_function(0, 0) == 0
