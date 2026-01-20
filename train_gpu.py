#!/usr/bin/env python3
"""
使用方式：
    python train_gpu.py --scenario A --num_episodes 1000 
    python train_gpu.py --scenario B --num_episodes 1000 
    python train_gpu.py --scenario C --num_episodes 1000 
"""

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import json
import argparse
import time
import random
import torch
import numpy as np
from datetime import datetime

from configs.config import RPSConfig, SCENARIOS
from envs import VectorizedRPSEnv
from agents import MAPPO_GPU, GPURolloutBuffer, PowerEnterpriseAgentV4


class GPUScenarioTrainer:
    """GPU场景训练器"""
    
    def __init__(self, scenario_name: str, num_episodes: int = 100):
       
        self.scenario_name = scenario_name
        self.num_episodes = num_episodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_parallel_envs = 1  # 固定为单环境
        
        # 设置随机种子为42（固定）
        self.seed = 42
        self._set_seeds(self.seed)
        
        # 检查场景有效性
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")
        
        print(f"[GPU Trainer] Device: {self.device}")
        print(f"[GPU Trainer] Scenario: {scenario_name} ({SCENARIOS[scenario_name]['description']})")
        print(f"[GPU Trainer] Seed: {self.seed}")
        
    def build_config(self) -> RPSConfig:
        """构建训练配置"""
        scenario_cfg = SCENARIOS[self.scenario_name]
        
        config = RPSConfig()
        # 环境配置
        config.environment.num_agents = config.environment.num_agents  # 使用config默认值(100)
        config.environment.max_periods = 11  
        config.environment.base_year = 2020
        config.environment.include_caq_market = scenario_cfg['include_caq_market']
        config.environment.include_tgc_market = scenario_cfg['include_tgc_market']
        
        # 政策配置
        config.policy.fine_per_unit = scenario_cfg['fine_per_unit']
        config.policy.reward_per_unit = scenario_cfg['reward_per_unit']
        
        # 训练配置 (GPU优化)
        config.training.rollout_length = 2048
        config.training.hidden_dims = [256, 256]  # 更大网络
        config.training.batch_size = 128
        config.training.num_epochs = 10
        config.training.lr_actor = 3e-4
        config.training.lr_critic = 3e-4
        config.training.gamma = 0.99
        config.training.gae_lambda = 0.95
        config.training.clip_epsilon = 0.2
        config.training.value_loss_coef = 0.5
        config.training.entropy_coef = 0.15  # 提高探索系数从0.01到0.15
        config.training.max_grad_norm = 0.5
        
        # GPU配置
        config.gpu.num_parallel_envs = self.num_parallel_envs
        config.gpu.use_mixed_precision = torch.cuda.is_available()  # 如果有CUDA则启用混合精度
        config.gpu.pin_memory = True
        
        return config
    
    def _set_seeds(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train(self):
        """执行训练"""
        config = self.build_config()
        
        # 创建agent manager
        agent_manager = PowerEnterpriseAgentV4(
            agent_config=config.agent,
            env_config=config.environment,
            market_config=config.market,
            gpu_config=config.gpu,
        )
        
        # 创建并行环境
        env = VectorizedRPSEnv(
            num_envs=self.num_parallel_envs,
            num_agents=config.environment.num_agents,
            max_periods=config.environment.max_periods,
            base_year=config.environment.base_year,
            device=self.device,
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
            num_envs=self.num_parallel_envs,
            hidden_dims=config.training.hidden_dims,
            lr_actor=config.training.lr_actor,
            lr_critic=config.training.lr_critic,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            clip_epsilon=config.training.clip_epsilon,
            value_loss_coef=config.training.value_loss_coef,
            entropy_coef=config.training.entropy_coef,
            max_grad_norm=config.training.max_grad_norm,
            use_attention=config.training.use_attention,
            use_mixed_precision=config.gpu.use_mixed_precision,
            device=self.device,
        )
        
        # 创建buffer
        buffer = GPURolloutBuffer(
            buffer_size=config.environment.max_periods,
            num_envs=self.num_parallel_envs,
            num_agents=config.environment.num_agents,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            global_state_dim=env.global_state_dim,
            device=self.device,
        )
        
        # 训练历史
        episode_rewards = []
        policy_losses = []
        value_losses = []
        
        
        reward_mean = 0.0
        reward_var = 1.0
        reward_count = 0
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            ep_start = time.time()
            
            # 重置环境
            obs, global_state = env.reset()
            buffer.reset()
            
            episode_reward = torch.zeros(self.num_parallel_envs, device=self.device)
            
            # 运行一个完整episode
            for step in range(config.environment.max_periods):
                # 获取动作
                with torch.no_grad():
                    actions, log_probs, values = agent.get_action(obs, global_state)
                
                # 环境步进
                next_obs, next_global_state, rewards, dones, infos = env.step(actions)
                
                # Running mean/std 归一化 reward
                batch_mean = rewards.mean().item()
                batch_std = rewards.std().item()
                batch_count = rewards.numel()
                
                # 更新全局的mean和var（Welford在线算法）
                delta = batch_mean - reward_mean
                reward_mean += delta * batch_count / (reward_count + batch_count)
                reward_var = (reward_var + (batch_std ** 2)) * reward_count / (reward_count + batch_count) if reward_count > 0 else batch_std ** 2
                reward_count += batch_count
                
                # 归一化reward
                eps = 1e-8
                normalized_rewards = (rewards - reward_mean) / (np.sqrt(reward_var) + eps)
                
                # 存储到buffer
                buffer.add(
                    obs=obs,
                    global_state=global_state,
                    action=actions,
                    reward=normalized_rewards,
                    done=dones,
                    log_prob=log_probs,
                    value=values,
                )
                
                episode_reward += rewards.sum(dim=1)  # 累积原始奖励用于显示
                
                obs = next_obs
                global_state = next_global_state
                
                if dones.any():
                    break
            
            # 计算GAE
            with torch.no_grad():
                _, _, last_value = agent.get_action(obs, global_state)
            buffer.compute_gae(last_value, config.training.gamma, config.training.gae_lambda)
            
            # 更新智能体
            metrics = agent.update(
                buffer,
                batch_size=config.training.batch_size,
                num_epochs=config.training.num_epochs,
            )
            
            # 记录
            avg_reward = episode_reward.mean().item()
            episode_rewards.append(avg_reward)
            policy_losses.append(metrics['policy_loss'])
            value_losses.append(metrics['value_loss'])
            
            ep_time = time.time() - ep_start
            
            # 每个episode都打印奖励
            if (episode + 1) % 1 == 0:
                print(f"  [{self.scenario_name}] Episode {episode+1:4d}/{self.num_episodes}: "
                      f"Reward={avg_reward:10.2f}")
            
            # 每100个episode保存一次中间检查点
            if (episode + 1) % 100 == 0:
                checkpoint_dir = f'./gpu_models/{self.scenario_name}_checkpoints'
                os.makedirs(checkpoint_dir, exist_ok=True)
                intermediate_checkpoint = os.path.join(
                    checkpoint_dir,
                    f'{self.scenario_name}_episode_{episode+1:04d}.pt'
                )
                agent.save(intermediate_checkpoint)
        
        total_time = time.time() - start_time
        
        # 保存结果
        results = {
            'episode_rewards': episode_rewards,
            'policy_losses': policy_losses,
            'value_losses': value_losses,
            'elapsed_time': total_time,
            'avg_reward': float(np.mean(episode_rewards)),
            'final_reward': float(episode_rewards[-1]),
            'timestamp': datetime.now().isoformat(),
        }
        
        # 创建检查点目录
        checkpoint_dir = f'./gpu_models/{self.scenario_name}_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存最终模型
        os.makedirs('./gpu_models', exist_ok=True)
        final_checkpoint = os.path.join(
            './gpu_models', 
            f'{self.scenario_name}_{self.num_episodes}ep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        )
        agent.save(final_checkpoint)
        
        # 保存训练元数据和奖励数据
        metadata_path = final_checkpoint.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存奖励数据供后续绘制
        rewards_path = final_checkpoint.replace('.pt', '_rewards.json')
        rewards_data = {
            'episode_rewards': episode_rewards,
            'policy_losses': [float(x) for x in policy_losses],
            'value_losses': [float(x) for x in value_losses],
        }
        with open(rewards_path, 'w') as f:
            json.dump(rewards_data, f, indent=2)
        
        
        
        return results


# 解析命令行参数
parser = argparse.ArgumentParser(description='GPU Scenario Training')
parser.add_argument('--scenario', type=str, default='A', 
                    help='Scenario name (A, A1-A7, B, C)')
parser.add_argument('--num_episodes', type=int, default=100, 
                    help='Number of training episodes (default: 100 for quick debugging)')

args = parser.parse_args()

# 创建训练器并执行
trainer = GPUScenarioTrainer(
    scenario_name=args.scenario,
    num_episodes=args.num_episodes,
)

results = trainer.train()

