# LLM-Enhanced UAV Path Planning Initialization

This project presents an implementation of large language model (LLM) powered intelligent initial population generation for 3D multi-UAV path planning, accessed via the OpenRouter API. It includes complete prompt templates, LLM invocation encapsulation, and response parsing logic, and can be directly integrated into related research or engineering projects.

## Features
- Support for multiple LLM backends (GPT-4o, Claude, Llama, etc.) via OpenRouter
- Structured prompt design to guide LLMs in generating terrain-adaptive paths
- Robust JSON parsing and data validation pipeline
- Random fallback mechanism to ensure runtime stability when API calls fail

## Requirements
- Python 3.8+
- `openai>=1.0.0`
- `numpy`

Install dependencies:
```bash
pip install -r requirements.txt

## Quick Start
1. Install dependencies:
   pip install -r requirements.txt
2. Configure your OpenRouter API key in config.py.
3. Run the example to test the initialization pipeline:
   python example.py
