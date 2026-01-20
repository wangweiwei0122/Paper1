#!/usr/bin/env python3
"""
诊断脚本：验证奖励修复
这个脚本会运行一个小规模的环境步骤，输出奖励值的大小和分布
用于验证奖励不再为0的问题是否已解决
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import numpy as np
from configs.config import RPSConfig, SCENARIOS
from envs import VectorizedRPSEnv
from agents import PowerEnterpriseAgentV4

# 设置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scenario_name = 'A'
scenario_cfg = SCENARIOS[scenario_name]

print(f"[测试] 设备: {device}")
print(f"[测试] 场景: {scenario_name}")
print(f"[测试] 细节: {scenario_cfg['description']}")
print()

# 构建配置
config = RPSConfig()
config.environment.num_agents = 100
config.environment.max_periods = 11
config.environment.base_year = 2020
config.environment.include_caq_market = scenario_cfg['include_caq_market']
config.environment.include_tgc_market = scenario_cfg['include_tgc_market']
config.policy.fine_per_unit = scenario_cfg['fine_per_unit']
config.policy.reward_per_unit = scenario_cfg['reward_per_unit']
config.gpu.num_parallel_envs = 1
config.training.hidden_dims = [256, 256]

# 创建agent manager
agent_manager = PowerEnterpriseAgentV4(
    num_envs=1,
    agent_config=config.agent,
    env_config=config.environment,
    market_config=config.market,
    gpu_config=config.gpu,
)

# 创建环境
env = VectorizedRPSEnv(
    num_envs=1,
    num_agents=config.environment.num_agents,
    max_periods=config.environment.max_periods,
    base_year=config.environment.base_year,
    device=device,
    agent_manager=agent_manager,
    fine_per_unit=config.policy.fine_per_unit,
    reward_per_unit=config.policy.reward_per_unit,
    include_caq_market=config.environment.include_caq_market,
    include_tgc_market=config.environment.include_tgc_market,
)

print("[测试] 环境初始化完成")
print(f"[测试] 奖励配置:")
print(f"      - components: {env.reward_config.components}")
print(f"      - weights: {env.reward_config.weights}")
print(f"      - normalize: {env.reward_config.normalize}")
print(f"      - scale: {env.reward_config.scale}")
print()

# 获取初始观测
obs, global_state = env.reset()

# 运行几个step，输出奖励统计
print("[测试] 运行10个环境step，输出奖励统计：")
print("-" * 80)
print(f"{'Step':>5} {'Reward Mean':>15} {'Reward Std':>15} {'Reward Max':>15} {'Reward Min':>15}")
print("-" * 80)

for step_idx in range(10):
    # 采用随机动作
    actions = torch.randn(1, config.environment.num_agents, 1, device=device)
    
    # 执行环境step
    next_obs, next_global_state, rewards, dones, infos = env.step(actions)
    
    # 统计奖励
    reward_np = rewards.cpu().numpy()  # shape: (B, N)
    reward_mean = reward_np.mean()
    reward_std = reward_np.std()
    reward_max = reward_np.max()
    reward_min = reward_np.min()
    
    print(f"{step_idx:5d} {reward_mean:15.6e} {reward_std:15.6e} {reward_max:15.6e} {reward_min:15.6e}")
    
    obs = next_obs
    global_state = next_global_state

print("-" * 80)
print()
print("[诊断结果]")

# 检查是否仍然为0
if np.allclose(reward_mean, 0.0, atol=1e-8):
    print("❌ 奖励仍然为0（问题未解决）")
    print("   可能原因：")
    print("   1. scale因子仍然过小（需要进一步调整）")
    print("   2. 利润计算出错（收入/成本异常）")
    print("   3. 奖励聚合有其他问题")
else:
    print("✅ 奖励不为0（问题已解决）")
    print(f"   - 奖励均值: {reward_mean:.6e}")
    print(f"   - 奖励标准差: {reward_std:.6e}")
    print(f"   - 奖励范围: [{reward_min:.6e}, {reward_max:.6e}]")

# 输出合规性信息
print()
print("[合规率信息]")
print(f"  - compliance_rate (最后一步): {infos['compliance_rate'][0].item():.4f}")
print(f"  - 合规判断标准（新）: 实际绿电消费 >= RPS要求")
print()
