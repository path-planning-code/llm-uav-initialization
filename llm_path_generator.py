import openai
import numpy as np
import json
import re
from typing import Dict, List, Optional
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODELS, DEFAULT_MODEL


class LLMPathGenerator:
    """
    通过OpenRouter调用大语言模型，为多无人机生成智能初始路径种群。
    """

    def __init__(self, api_key: str = None, model_name: str = None, base_url: str = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model_name = model_name or DEFAULT_MODEL
        self.base_url = base_url or OPENROUTER_BASE_URL

        # 获取实际模型ID（例如 "openai/gpt-4o-mini"）
        self.model_id = OPENROUTER_MODELS.get(self.model_name, OPENROUTER_MODELS[DEFAULT_MODEL])

        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"LLM路径生成器初始化完成，使用模型: {self.model_name} ({self.model_id})")
        else:
            self.client = None
            print("警告: 未提供API密钥，LLM功能不可用")

    def generate_initial_population(
        self,
        terrain_features: Dict,
        num_drones: int,
        population_size: int,
        point_num: int,
        pos_bound: np.ndarray
    ) -> List[List[Dict]]:
        """
        生成整个初始种群（每个个体包含多架无人机的路径点）
        """
        if not self.client:
            return self._generate_random_population(population_size, num_drones, point_num, pos_bound)

        prompt = self._build_population_prompt(
            terrain_features, num_drones, population_size, point_num, pos_bound
        )

        try:
            print(f"通过OpenRouter调用 {self.model_name} 生成种群...")
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "你是一个路径规划专家，请返回纯JSON格式数据，不要包含其他文本。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            content = response.choices[0].message.content
            return self._parse_population_response(content, population_size, num_drones, point_num, pos_bound)
        except Exception as e:
            print(f"LLM调用失败: {e}，使用随机种群")
            return self._generate_random_population(population_size, num_drones, point_num, pos_bound)

    def _build_population_prompt(self, features, num_drones, population_size, point_num, pos_bound):
        """构建种群生成的Prompt模板"""
        prompt = f"""
作为多无人机路径规划专家，请为{population_size}个不同的路径方案（种群个体）生成初始路径。每个方案包含{num_drones}架无人机的路径，每架无人机的路径包含{point_num}个中间点。

地形特征分析：
- 地形类型：{features.get('terrain_type', '未知')}
- 高程范围：{features['terrain_statistics']['min_elevation']:.2f} ~ {features['terrain_statistics']['max_elevation']:.2f}
- 平均高程：{features['terrain_statistics']['mean_elevation']:.2f}
- 地形粗糙度：{features['terrain_statistics']['roughness']:.3f}
- 障碍物密度：{features['terrain_statistics']['obstacle_density']:.1%}
- 路径复杂度评分：{features['path_complexity']['complexity_score']:.2f}

起点坐标：{features['start_position']}
终点坐标：{features['goal_position']}
地图范围：{features['map_dimensions']}
无人机数量：{num_drones}
种群大小：{population_size}

请生成多样化的路径策略组合，包括：
1. 直接路径策略：近似直线，快速但可能有碰撞风险
2. 绕行策略：避开主要障碍，安全但路径较长
3. 高空飞行策略：提升高度避开复杂地形
4. 山谷跟随策略：利用低洼地形节省能量
5. 混合策略：结合多种方法的创新路径

请确保：
- 每个方案中的无人机路径保持安全距离
- 种群整体具有足够的多样性
- 路径适应地形特征

请以纯JSON格式返回结果，不要包含任何其他文本。结构如下：
{{
    "population": [
        {{
            "strategy": "方案1策略描述",
            "drones_paths": [
                {{
                    "x": [x1, x2, ..., x{point_num}],
                    "y": [y1, y2, ..., y{point_num}],
                    "z": [z1, z2, ..., z{point_num}]
                }},
                ... // {num_drones}个无人机的路径
            ]
        }},
        ... // {population_size}个方案
    ]
}}

坐标范围限制：
x: [{pos_bound[0, 0]}, {pos_bound[0, 1]}]
y: [{pos_bound[1, 0]}, {pos_bound[1, 1]}]
z: [{pos_bound[2, 0]}, {pos_bound[2, 1]}]

请确保所有坐标都在上述范围内，并且路径点数量准确。

重要：请只返回JSON格式数据，不要包含任何其他文本！
"""
        return prompt

    def _parse_population_response(self, response: str, population_size: int,
                                   num_drones: int, point_num: int,
                                   pos_bound: np.ndarray) -> List[List[Dict]]:
        """解析LLM返回的JSON数据，并进行验证和修复"""
        # 提取JSON部分
        json_str = self._extract_json(response)
        try:
            data = json.loads(json_str)
            population_data = data.get('population', [])
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}，使用随机种群")
            return self._generate_random_population(population_size, num_drones, point_num, pos_bound)

        validated_population = []
        for individual in population_data:
            if len(validated_population) >= population_size:
                break
            drones_paths = individual.get('drones_paths', [])
            validated_drones = []
            for path in drones_paths:
                if len(validated_drones) >= num_drones:
                    break
                try:
                    # 确保每个坐标轴都有数据，并裁剪到边界内
                    valid_path = {
                        'x': np.clip(self._to_numeric_array(path.get('x', [])), pos_bound[0, 0], pos_bound[0, 1]),
                        'y': np.clip(self._to_numeric_array(path.get('y', [])), pos_bound[1, 0], pos_bound[1, 1]),
                        'z': np.clip(self._to_numeric_array(path.get('z', [])), pos_bound[2, 0], pos_bound[2, 1])
                    }
                    # 调整点数
                    for key in ['x', 'y', 'z']:
                        arr = valid_path[key]
                        if len(arr) > point_num:
                            valid_path[key] = arr[:point_num]
                        elif len(arr) < point_num:
                            # 线性插值到所需点数
                            if len(arr) > 1:
                                old = np.linspace(0, 1, len(arr))
                                new = np.linspace(0, 1, point_num)
                                valid_path[key] = np.interp(new, old, arr)
                            else:
                                valid_path[key] = np.full(point_num, arr[0] if len(arr) > 0 else 0)
                    validated_drones.append(valid_path)
                except Exception as e:
                    print(f"解析单个路径出错: {e}，跳过")
                    continue
            # 补充缺失的无人机路径
            while len(validated_drones) < num_drones:
                validated_drones.append(self._generate_random_path(point_num, pos_bound))
            validated_population.append(validated_drones)

        # 补充不足的种群个体
        while len(validated_population) < population_size:
            validated_population.append(self._generate_random_individual(num_drones, point_num, pos_bound))

        return validated_population[:population_size]

    def _extract_json(self, text: str) -> str:
        """从可能包含额外文本的响应中提取JSON字符串"""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("未找到有效的JSON对象")
        json_str = text[start:end]
        # 常见修复：单引号转双引号，键名加引号
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str

    def _to_numeric_array(self, data):
        """将输入转换为numpy float数组"""
        if isinstance(data, (list, np.ndarray)):
            try:
                return np.array([float(x) for x in data])
            except:
                return np.array([])
        elif isinstance(data, (int, float)):
            return np.array([float(data)])
        else:
            return np.array([])

    def _generate_random_population(self, population_size, num_drones, point_num, pos_bound):
        return [self._generate_random_individual(num_drones, point_num, pos_bound) for _ in range(population_size)]

    def _generate_random_individual(self, num_drones, point_num, pos_bound):
        return [self._generate_random_path(point_num, pos_bound) for _ in range(num_drones)]

    def _generate_random_path(self, point_num, pos_bound):
        return {
            'x': np.random.uniform(pos_bound[0, 0], pos_bound[0, 1], point_num),
            'y': np.random.uniform(pos_bound[1, 0], pos_bound[1, 1], point_num),
            'z': np.random.uniform(pos_bound[2, 0], pos_bound[2, 1], point_num)
        }
