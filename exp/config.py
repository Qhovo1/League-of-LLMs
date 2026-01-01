"""
Config
"""

# API Configuration
API_KEY = "" 
API_BASE = ""

# Experiment Configuration
STREAMING = True
DEFAULT_TEMPERATURE = 1.0

# Model Configuration - Define all models to be tested here
MODELS = [
    {
        "name": "gpt-4.1",
        "description": "gpt-4.1 Model",
        "api_key": API_KEY
    },
    {
        "name": "o3-mini",
        "description": "o3-mini Model",
        "api_key": API_KEY
    },
    {
        "name": "o1",
        "description": "o1 Model",
        "api_key": API_KEY,
        "streaming": False  # Disable streaming output for o1 model
    },
    {
        "name": "claude-3-7-sonnet-20250219",
        "description": "claude-3-7 Model",
        "api_key": API_KEY
    },
    {
        "name": "deepseek-r1",
        "description": "deepseek-r1 Model",
        "api_key": API_KEY
    },
    {
        "name": "deepseek-v3",
        "description": "deepseek-v3 Model",
        "api_key": API_KEY
    },
    {
        "name": "qwen2.5-max",
        "description": "qwen2.5-max Model",
        "api_key": API_KEY
    },
    {
        "name": "gemini-2.5-pro-exp-03-25",
        "description": "gemini-2.5-pro-exp Model",
        "api_key": API_KEY
    }
]

# Experiment Output Configuration
RESULTS_DIR = "results" 