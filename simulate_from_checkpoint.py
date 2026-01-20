#!/usr/bin/env python3
"""
从特定的中间checkpoint运行仿真
使用方法：
    python simulate_from_checkpoint.py --scenario A --episode 500
    python simulate_from_checkpoint.py --scenario B --episode 250
"""

import sys
import os

# 添加当前目录到Python路径以支持相对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from configs.config import RPSConfig, SCENARIOS, get_scenario_config
from envs import VectorizedRPSEnv
from agents import MAPPO_GPU, PowerEnterpriseAgentV4


def find_checkpoint(scenario_name: str, episode: int) -> str:
    """查找指定轮次的checkpoint"""
    checkpoint_path = f'./gpu_models/{scenario_name}_checkpoints/{scenario_name}_episode_{episode:04d}.pt'
    
    if not os.path.exists(checkpoint_path):
        available = list(Path(f'./gpu_models/{scenario_name}_checkpoints').glob('*.pt'))
        if not available:
            raise FileNotFoundError(
                f"No checkpoint found for scenario {scenario_name}, episode {episode}. "
                f"Check if directory './gpu_models/{scenario_name}_checkpoints' exists."
            )
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available checkpoints:\n" +
            "\n".join([str(p.name) for p in sorted(available)])
        )
    
    return checkpoint_path


# 解析命令行参数
parser = argparse.ArgumentParser(
    description='Run simulation from specific checkpoint',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python simulate_from_checkpoint.py --scenario A --episode 500
  python simulate_from_checkpoint.py --scenario B --episode 250
  python simulate_from_checkpoint.py --scenario C --episode 100
    """)

parser.add_argument('--scenario', type=str, default='A', 
                    help='Scenario name (A, A1-A7, B, C)')
parser.add_argument('--episode', type=int, required=True,
                    help='Episode number to load (e.g., 500, 1000)')

args = parser.parse_args()

scenario_name = args.scenario
episode_num = args.episode

# 验证场景有效性
if scenario_name not in SCENARIOS:
    raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")

print(f"[Checkpoint Simulator] Scenario: {scenario_name}")
print(f"[Checkpoint Simulator] Episode: {episode_num}")

# 查找checkpoint
checkpoint_path = find_checkpoint(scenario_name, episode_num)
print(f"[Checkpoint Simulator] Checkpoint: {checkpoint_path}")

# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Checkpoint Simulator] Device: {device}")

# 构建config
scenario_cfg = get_scenario_config(scenario_name)
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

# 创建和加载agent
agent = MAPPO_GPU(
    obs_dim=env.obs_dim,
    action_dim=env.action_dim,
    global_state_dim=env.global_state_dim,
    num_agents=config.environment.num_agents,
    num_envs=1,
    hidden_dims=config.training.hidden_dims,
    device=device,
)

# 加载checkpoint
agent.load(checkpoint_path)
print(f"[Checkpoint Simulator] Model loaded successfully")

# 执行仿真
step_data = []  # 市场层面数据
agent_level_data = []  # 企业层面数据

obs, global_state = env.reset()

print(f"[Checkpoint Simulator] Starting simulation...")
for step_idx in range(config.environment.max_periods):
    # 获取确定性动作
    with torch.no_grad():
        actions, _, _ = agent.get_action(obs, global_state, deterministic=True)

    # 执行环境
    next_obs, next_global_state, rewards, dones, infos = env.step(actions)

    # 提取数据
    year = int(infos['year'][0].item())

    # 市场交易数据
    record = {
        'year': year,
        'caq_price': float(infos['caq_price'][0].item()) if config.environment.include_caq_market else np.nan,
        'tgc_price': float(infos['tgc_price'][0].item()) if config.environment.include_tgc_market else np.nan,
        'compliance_rate': float(infos['compliance_rate'][0].item()),
        'mean_re_share': float(infos['mean_re_share'][0].item()),
        'std_re_share': float(infos['std_re_share'][0].item()),
        'num_caq_buyers': int(infos['num_caq_buyers'][0].item()) if config.environment.include_caq_market else 0,
        'num_caq_sellers': int(infos['num_caq_sellers'][0].item()) if config.environment.include_caq_market else 0,
        'caq_volume': float(infos['caq_volume'][0].item()) if config.environment.include_caq_market else 0.0,
        'num_tgc_buyers': int(infos['num_tgc_buyers'][0].item()) if config.environment.include_tgc_market else 0,
        'num_tgc_sellers': int(infos['num_tgc_sellers'][0].item()) if 'num_tgc_sellers' in infos and config.environment.include_tgc_market else 0,
        'tgc_volume': float(infos['tgc_volume'][0].item()) if config.environment.include_tgc_market else 0.0,
        'profit_mean': float(infos['profit_mean'][0].item()) if 'profit_mean' in infos else 0.0,
        'profit_std': float(infos['profit_std'][0].item()) if 'profit_std' in infos else 0.0,
    }
    step_data.append(record)

    # 企业层面数据
    demands = env.agent_manager.demand[0].cpu().numpy()
    re_shares = env.agent_re_share[0].cpu().numpy()
    quota_balances = env.agent_quota_balance[0].cpu().numpy()
    profits = env.last_profit_per_agent[0].cpu().numpy() if env.last_profit_per_agent is not None else None
    
    for agent_id in range(config.environment.num_agents):
        agent_record = {
            'year': year,
            'agent_id': agent_id,
            'demand': float(demands[agent_id]),
            're_share': float(re_shares[agent_id]),
            'quota_balance': float(quota_balances[agent_id]),
            'compliance': 1 if quota_balances[agent_id] >= 0 else 0,
        }
        if profits is not None:
            agent_record['profit'] = float(profits[agent_id])
        agent_level_data.append(agent_record)

    obs = next_obs
    global_state = next_global_state

# 分类企业规模
agent_df = pd.DataFrame(agent_level_data)
initial_year_data = agent_df[agent_df['year'] == 2020]
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

# 保存数据到CSV（包含轮次标识）
output_suffix = f"{scenario_name}_ep{episode_num:04d}"
df_market = pd.DataFrame(step_data)
df_market.to_csv(f'{output_suffix}_market.csv', index=False)

df_agents = agent_df
df_agents.to_csv(f'{output_suffix}_agents.csv', index=False)

print(f"[Checkpoint Simulator] Simulation complete!")
print(f"[Checkpoint Simulator] Market data saved to: {output_suffix}_market.csv")
print(f"[Checkpoint Simulator] Agent data saved to: {output_suffix}_agents.csv")
