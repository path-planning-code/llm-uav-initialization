# LLM-Enhanced UAV Path Planning Initialization

本项目展示如何利用大语言模型（通过OpenRouter API）为多无人机三维路径规划生成智能初始种群。包含完整的Prompt模板、LLM调用封装和响应解析逻辑，可直接集成到您的研究或项目中。

## 特性
- 支持多种LLM模型（GPT-4o, Claude, Llama等）
- 结构化Prompt设计，引导LLM生成地形自适应路径
- 鲁棒的JSON解析与数据验证
- 随机回退机制，保证API调用失败时仍可运行

## 依赖
- Python 3.8+
- `openai>=1.0.0`
- `numpy`

安装依赖：
```bash
pip install -r requirements.txt
