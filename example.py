import numpy as np
from llm_path_generator import LLMPathGenerator

def main():
    # 模拟地形特征数据（实际使用时请替换为真实地形分析结果）
    terrain_features = {
        'terrain_type': '丘陵地形',
        'terrain_statistics': {
            'min_elevation': 10.0,
            'max_elevation': 85.0,
            'mean_elevation': 42.5,
            'roughness': 0.25,
            'obstacle_density': 0.35
        },
        'path_complexity': {
            'complexity_score': 25.6
        },
        'start_position': [1, 1, 1],
        'goal_position': [100, 100, 80],
        'map_dimensions': [100, 100, 100]
    }

    # 参数设置
    num_drones = 5          # 无人机数量
    population_size = 20    # 种群大小（路径方案数）
    point_num = 3           # 每条路径的中间航点数
    pos_bound = np.array([[0, 100], [0, 100], [0, 100]])  # [xmin,xmax; ymin,ymax; zmin,zmax]

    # 初始化LLM路径生成器（默认使用config中的模型）
    generator = LLMPathGenerator()

    # 生成初始种群
    population = generator.generate_initial_population(
        terrain_features=terrain_features,
        num_drones=num_drones,
        population_size=population_size,
        point_num=point_num,
        pos_bound=pos_bound
    )

    print(f"成功生成 {len(population)} 个路径方案")
    for i, individual in enumerate(population[:2]):  # 只打印前两个示例
        print(f"\n方案 {i+1}:")
        for j, drone_path in enumerate(individual):
            print(f"  无人机 {j+1}: x={drone_path['x']}, y={drone_path['y']}, z={drone_path['z']}")

if __name__ == "__main__":
    main()
