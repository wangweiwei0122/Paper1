#!/usr/bin/env python3
import sys
import os

# 添加当前目录到Python路径以支持相对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from configs.config import RPSConfig, SCENARIOS, get_scenario_config
from envs import VectorizedRPSEnv, RewardRegistry, RewardContext
from agents import MAPPO_GPU, PowerEnterpriseAgentV4

# 设置参数
scenario_name = 'C'
output_dir = './results/'
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# 查找最新的模型文件
def find_latest_model(scenario_name: str) -> str:
    base_dir = Path("./results/rps_marl_v4")
    if not base_dir.exists():
        base_dir = Path("../results/rps_marl_v4")
    model_dirs = []
    for pattern in [f"scenario_{scenario_name}_*", f"{scenario_name}_*"]:
        for d in base_dir.glob(pattern):
            if (d / "checkpoint_final.pt").exists():
                model_dirs.append(d)
    if not model_dirs:
        raise FileNotFoundError(f"No models found for scenario {scenario_name}")
    latest_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
    return str(latest_dir / "checkpoint_final.pt")

try:
    checkpoint_path = find_latest_model(scenario_name)
except FileNotFoundError:
    checkpoint_path = None

os.makedirs(output_dir, exist_ok=True)

# 构建config
scenario_cfg = get_scenario_config(scenario_name)
config = RPSConfig()
config.environment.num_agents = 100
config.environment.max_periods = 11  # 2020-2030 (11个周期)
config.environment.base_year = 2020
config.environment.include_caq_market = scenario_cfg['include_caq_market']
config.environment.include_tgc_market = scenario_cfg['include_tgc_market']
config.policy.fine_per_unit = scenario_cfg['fine_per_unit']
config.policy.reward_per_unit = scenario_cfg['reward_per_unit']
config.gpu.num_parallel_envs = 1  # 使用单个环境
config.training.hidden_dims = [256, 256]

# 创建agent manager
agent_manager = PowerEnterpriseAgentV4(
    num_envs=1,  # 使用单个环境
    agent_config=config.agent,
    env_config=config.environment,
    market_config=config.market,
    gpu_config=config.gpu,
)

# 创建环境
env = VectorizedRPSEnv(
    num_envs=1,  # 使用单个环境
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

# 创建智能体
agent = MAPPO_GPU(
    obs_dim=env.obs_dim,
    action_dim=env.action_dim,
    global_state_dim=env.global_state_dim,
    num_agents=config.environment.num_agents,
    num_envs=1,  # 使用单个环境
    hidden_dims=config.training.hidden_dims,
    device=device,
)

# 加载模型
agent.load(checkpoint_path)
print(f"[Scenario {scenario_name}] Model loaded from {checkpoint_path}")

# 执行仿真
step_data = []  # 市场层面数据
agent_level_data = []  # 企业层面数据

obs, global_state = env.reset()

for step_idx in range(config.environment.max_periods):
    # 获取确定性动作
    with torch.no_grad():
        actions, _, _ = agent.get_action(obs, global_state, deterministic=True)

    # 执行环境
    next_obs, next_global_state, rewards, dones, infos = env.step(actions)

    # 提取数据
    year = int(infos['year'][0].item())

    # C类场景数据（仅TGC市场，无CAQ）
    record = {
        'year': year,
        # TGC市场价格（C类场景必有）
        'tgc_price': float(infos['tgc_price'][0].item()),
        # 政策指标
        'compliance_rate': float(infos['compliance_rate'][0].item()),
        'mean_re_share': float(infos['mean_re_share'][0].item()),
        'std_re_share': float(infos['std_re_share'][0].item()),
        # TGC交易指标
        'num_tgc_buyers': int(infos['num_tgc_buyers'][0].item()),
        'num_tgc_sellers': int(infos['num_tgc_sellers'][0].item()) if 'num_tgc_sellers' in infos else 0,
        'tgc_volume': float(infos['tgc_volume'][0].item()),
        # 利润统计
        'profit_mean': float(infos['profit_mean'][0].item()) if 'profit_mean' in infos else 0.0,
        'profit_std': float(infos['profit_std'][0].item()) if 'profit_std' in infos else 0.0,
    }

    step_data.append(record)

    # 【企业层面数据】用于异质性分析
    demands = env.agent_manager.demand[0].cpu().numpy()  # (N,) 100个企业的需求量
    re_shares = env.agent_re_share[0].cpu().numpy()  # (N,) RE份额
    quota_balances = env.agent_quota_balance[0].cpu().numpy()  # (N,) 配额余额
    # 【新增】获取每个agent的未归一化利润
    profits = env.last_profit_per_agent[0].cpu().numpy() if env.last_profit_per_agent is not None else None
    
    for agent_id in range(100):
        agent_record = {
            'year': year,
            'agent_id': agent_id,
            'demand': float(demands[agent_id]),
            're_share': float(re_shares[agent_id]),
            'quota_balance': float(quota_balances[agent_id]),
            'compliance': 1 if quota_balances[agent_id] >= 0 else 0,
        }
        # 【新增】添加利润数据（如果可用）
        if profits is not None:
            agent_record['profit'] = float(profits[agent_id])
        agent_level_data.append(agent_record)

    obs = next_obs
    global_state = next_global_state


agent_df = pd.DataFrame(agent_level_data)
initial_year_data = agent_df[agent_df['year'] == 2020]  # 仅2020年数据
demand_p25 = initial_year_data['demand'].quantile(0.25)
demand_p75 = initial_year_data['demand'].quantile(0.75)

def classify_scale(demand):
    if demand <= demand_p25:
        return 'small'
    elif demand > demand_p75:
        return 'large'
    else:
        return 'medium'

agent_df['enterprise_scale'] = agent_df['demand'].apply(classify_scale)

# 保存数据到CSV
df_market = pd.DataFrame(step_data)
df_market.to_csv('C_market.csv', index=False)

df_agents = agent_df
df_agents.to_csv('C_agents.csv', index=False)




    