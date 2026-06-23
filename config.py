# OpenRouter API 配置
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx"  # 请替换为您的密钥
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 支持的模型映射（可根据需要增删）
OPENROUTER_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
}

# 默认使用的模型
DEFAULT_MODEL = "gpt-4o-mini"