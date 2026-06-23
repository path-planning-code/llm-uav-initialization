import openai
import numpy as np
import json
import datetime
from typing import Dict
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODELS, DEFAULT_MODEL


class LLMTerrainAnalyzer:
    """
    使用LLM分析地形特征并生成路径建议（展示LLM的扩展应用）
    """

    def __init__(self, api_key=None, model_name=None, base_url=None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model_name = model_name or DEFAULT_MODEL
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.model_id = OPENROUTER_MODELS.get(self.model_name, OPENROUTER_MODELS[DEFAULT_MODEL])
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def analyze_terrain(self, terrain_stats: Dict, start: np.ndarray, goal: np.ndarray) -> str:
        """
        调用LLM生成地形分析报告和路径建议
        """
        prompt = self._build_analysis_prompt(terrain_stats, start, goal)
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM分析失败: {e}"

    def _build_analysis_prompt(self, stats, start, goal):
        return f"""
请根据以下地形数据，为无人机路径规划提供简要建议：
- 高程范围: {stats['min_elevation']:.1f} ~ {stats['max_elevation']:.1f}
- 平均高程: {stats['mean_elevation']:.1f}
- 地形粗糙度: {stats['roughness']:.3f}
- 障碍物密度: {stats['obstacle_density']:.1%}
- 起点: {start.tolist()}
- 终点: {goal.tolist()}

请输出3-5条关键建议，包括飞行高度、绕行策略等。
"""
