#!/bin/bash
export HA_TOKEN="mock_token"
export PYTHONPATH=$PYTHONPATH:$(pwd) # Add current directory to PYTHONPATH
pytest --cov=. test/test_ygg_adapter.py      # Path to the test file relative to current directory