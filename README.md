# LLM-Enhanced Initial Waypoint Generation for 3D Multi-UAV Path Planning

> This is an anonymous repository for double-blind peer review. The official version with full author information will be released upon paper acceptance.

This repository provides the reference implementation of the LLM-enhanced terrain-aware initial population generation method described in Section IV-A of the submitted paper. The pipeline is accessed via the OpenRouter API, with complete prompt templates, LLM invocation encapsulation, and response parsing logic, which can be directly integrated into related research or engineering projects.

## Features
- Support for multiple LLM backends (GPT-4o, Claude, Llama, etc.) via OpenRouter
- Structured prompt design to guide LLMs in generating terrain-adaptive paths
- Robust JSON parsing and data validation pipeline
- Random fallback mechanism to ensure runtime stability when API calls fail

## Requirements
- Python 3.8+
- `openai>=1.0.0`
- `numpy`

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Configure your OpenRouter API key in config.py.
3. Run the example to test the initialization pipeline:
    ```bash
   python example.py
