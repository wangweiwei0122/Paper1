#!/usr/bin/env python3
"""
奖励曲线绘制脚本
用法: python plot_reward_from_checkpoint.py --scenario A --episode 100
     输入第N轮的模型，自动运行一次完整仿真（11期），绘制奖励曲线
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
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
        available_episodes = sorted([int(p.stem.split('_')[-1]) for p in available])
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available episodes: {available_episodes}"
        )
    
    return checkpoint_path


# 解析命令行参数
parser = argparse.ArgumentParser(description='Plot reward from checkpoint')
parser.add_argument('--scenario', type=str, default='A', help='Scenario name (A, B, C, ...)')
parser.add_argument('--episode', type=int, default=100, help='Episode number to load')

args = parser.parse_args()

scenario_name = args.scenario
episode_num = args.episode

print(f"[绘制工具] Scenario: {scenario_name}, Episode: {episode_num}")

# 查找checkpoint
try:
    checkpoint_path = find_checkpoint(scenario_name, episode_num)
    print(f"[绘制工具] Checkpoint: {checkpoint_path}")
except FileNotFoundError as e:
    print(f"❌ 错误: {e}")
    sys.exit(1)

# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[绘制工具] Device: {device}")

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
print(f"[绘制工具] Model loaded successfully")

# 执行仿真，收集奖励数据
rewards_per_period = []  # 每期的平均奖励
compliance_rates = []     # 每期的合规率
re_shares = []           # 每期的绿电比例
years = []

obs, global_state = env.reset()

print(f"[绘制工具] Running simulation for {config.environment.max_periods} periods...")
for step_idx in range(config.environment.max_periods):
    # 获取确定性动作
    with torch.no_grad():
        actions, _, _ = agent.get_action(obs, global_state, deterministic=True)
    
    # 执行环境
    next_obs, next_global_state, rewards, dones, infos = env.step(actions)
    
    # 收集数据
    year = int(infos['year'][0].item())
    avg_reward = rewards[0].sum().item()  # 这一期所有agent的总奖励
    compliance_rate = float(infos['compliance_rate'][0].item())
    mean_re_share = float(infos['mean_re_share'][0].item())
    
    rewards_per_period.append(avg_reward)
    compliance_rates.append(compliance_rate)
    re_shares.append(mean_re_share)
    years.append(year)
    
    print(f"  Period {step_idx+1:2d} (Year {year}): Reward={avg_reward:10.2f}, "
          f"Compliance={compliance_rate:5.2%}, RE_share={mean_re_share:5.2%}")
    
    obs = next_obs
    global_state = next_global_state

print(f"[绘制工具] Simulation complete!")

# 绘制结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Scenario {scenario_name} - Episode {episode_num} - Training Results', fontsize=16, fontweight='bold')

# 1. 奖励曲线
ax = axes[0, 0]
ax.plot(years, rewards_per_period, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Total Reward (All Agents)', fontsize=11)
ax.set_title('Reward per Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(years)

# 2. 合规率曲线
ax = axes[0, 1]
ax.plot(years, compliance_rates, 's-', linewidth=2, markersize=8, color='#A23B72')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Compliance Rate', fontsize=11)
ax.set_title('Compliance Rate (RPS Requirement)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(years)
ax.set_ylim([0, 1.05])

# 3. 绿电比例曲线
ax = axes[1, 0]
ax.plot(years, re_shares, '^-', linewidth=2, markersize=8, color='#F18F01')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Renewable Energy Share (%)', fontsize=11)
ax.set_title('Mean RE Share (All Agents)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(years)
ax.set_ylim([0, 1.05])

# 4. 统计信息
ax = axes[1, 1]
ax.axis('off')

# 计算统计量
avg_reward = np.mean(rewards_per_period)
avg_compliance = np.mean(compliance_rates)
avg_re_share = np.mean(re_shares)
final_reward = rewards_per_period[-1]
final_compliance = compliance_rates[-1]
final_re_share = re_shares[-1]

stats_text = f"""
SUMMARY STATISTICS

Scenario: {scenario_name}
Episode: {episode_num}
Device: {device}

Reward:
  • Mean: {avg_reward:.2f}
  • Final: {final_reward:.2f}
  • Range: [{min(rewards_per_period):.2f}, {max(rewards_per_period):.2f}]

Compliance Rate:
  • Mean: {avg_compliance*100:.2f}%
  • Final: {final_compliance*100:.2f}%
  
RE Share:
  • Mean: {avg_re_share*100:.2f}%
  • Final: {final_re_share*100:.2f}%

Periods: {config.environment.max_periods}
Agents: {config.environment.num_agents}
Years: {years[0]}-{years[-1]}
"""

ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图片
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{scenario_name}_episode_{episode_num:04d}_reward.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[绘制工具] Plot saved to: {output_path}")

plt.show()

# 可选：保存数据到CSV
import pandas as pd

data_df = pd.DataFrame({
    'year': years,
    'reward': rewards_per_period,
    'compliance_rate': compliance_rates,
    're_share': re_shares,
})

csv_path = os.path.join(output_dir, f'{scenario_name}_episode_{episode_num:04d}_metrics.csv')
data_df.to_csv(csv_path, index=False)
print(f"[绘制工具] Data saved to: {csv_path}")
