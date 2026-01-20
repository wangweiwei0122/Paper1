
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.cuda.amp import GradScaler
from typing import Dict, List, Tuple, Optional
import numpy as np


class ActorNetwork(nn.Module):
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_init: float = -0.5,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build MLP layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Output heads
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # 输出层小初始化
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        features = self.feature_net(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mean, std = self(obs)
        
        if deterministic:
            # Deterministic: use mean, apply tanh
            action = torch.tanh(mean)
            # Log prob for tanh(mean) - Jacobian correction simplified
            log_prob = torch.zeros(obs.shape[:-1] + (1,), device=obs.device)
            entropy = torch.zeros(obs.shape[:-1] + (1,), device=obs.device)
        else:
            # Stochastic: sample from Gaussian, then tanh
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            
            # Log probability with Jacobian correction for tanh squashing
            # log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
            log_prob_raw = dist.log_prob(raw_action)
            log_prob_correction = torch.log(1 - action.pow(2) + 1e-6)
            log_prob = (log_prob_raw - log_prob_correction).sum(dim=-1, keepdim=True)
            
            # Entropy
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        mean, std = self(obs)
        
        # Inverse tanh to get raw action
        raw_action = torch.atanh(actions.clamp(-0.999, 0.999))
        
        dist = torch.distributions.Normal(mean, std)
        log_prob_raw = dist.log_prob(raw_action)
        log_prob_correction = torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = (log_prob_raw - log_prob_correction).sum(dim=-1, keepdim=True)
        
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class CriticNetworkWithAttention(nn.Module):

    
    def __init__(
        self,
        global_state_dim: int,
        num_agents: int,
        hidden_dims: List[int] = [256, 256],
        attention_heads: int = 4,
        attention_dim: int = 64,
    ):
        super().__init__()
        
        self.global_state_dim = global_state_dim
        self.num_agents = num_agents
        
        # Agent embedding
        self.agent_embedding = nn.Linear(global_state_dim // num_agents + 1, attention_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        
        # Value head
        value_input_dim = attention_dim + global_state_dim
        layers = []
        prev_dim = value_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.value_net = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
    
        B = global_state.shape[0]
        
        # Split global state into agent-wise features (simplified)
        # For attention, create pseudo-agent tokens
        agent_dim = self.global_state_dim // self.num_agents
        if agent_dim * self.num_agents < self.global_state_dim:
            # Pad
            pad_size = agent_dim * self.num_agents - self.global_state_dim + agent_dim
            global_state_padded = F.pad(global_state, (0, pad_size))
        else:
            global_state_padded = global_state
            
        # Reshape to (B, num_agents, agent_dim)
        try:
            agent_features = global_state_padded[:, :agent_dim * self.num_agents].reshape(B, self.num_agents, -1)
        except:
            # Fallback: use global state directly
            agent_features = global_state.unsqueeze(1).expand(B, self.num_agents, -1)[:, :, :agent_dim]
        
        # Add agent index embedding
        agent_idx = torch.arange(self.num_agents, device=global_state.device).float()
        agent_idx = agent_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1) / self.num_agents
        agent_features = torch.cat([agent_features[..., :agent_dim], agent_idx], dim=-1)
        
        # Embed
        agent_embed = self.agent_embedding(agent_features)  # (B, N, attention_dim)
        
        # Self-attention
        attn_out, _ = self.attention(agent_embed, agent_embed, agent_embed)  # (B, N, attention_dim)
        
        # Pool attention output
        attn_pooled = attn_out.mean(dim=1)  # (B, attention_dim)
        
        # Concatenate with global state
        combined = torch.cat([attn_pooled, global_state], dim=-1)
        
        # Value prediction
        value = self.value_net(combined)
        
        return value


class GPURolloutBuffer:

    
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        
        # Pre-allocate tensors
        self.observations = torch.zeros(
            buffer_size, num_envs, num_agents, obs_dim, 
            device=device, dtype=torch.float32
        )
        self.global_states = torch.zeros(
            buffer_size, num_envs, global_state_dim,
            device=device, dtype=torch.float32
        )
        self.actions = torch.zeros(
            buffer_size, num_envs, num_agents, action_dim,
            device=device, dtype=torch.float32
        )
        self.rewards = torch.zeros(
            buffer_size, num_envs, num_agents,
            device=device, dtype=torch.float32
        )
        self.dones = torch.zeros(
            buffer_size, num_envs,
            device=device, dtype=torch.bool
        )
        self.log_probs = torch.zeros(
            buffer_size, num_envs, num_agents, 1,
            device=device, dtype=torch.float32
        )
        self.values = torch.zeros(
            buffer_size, num_envs, 1,
            device=device, dtype=torch.float32
        )
        
        # Computed during finalization
        self.advantages = torch.zeros(
            buffer_size, num_envs, num_agents,
            device=device, dtype=torch.float32
        )
        self.returns = torch.zeros(
            buffer_size, num_envs, num_agents,
            device=device, dtype=torch.float32
        )
        
        self.ptr = 0
        self.full = False
        
    def add(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ):
        """添加一步数据"""
        self.observations[self.ptr] = obs
        self.global_states[self.ptr] = global_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0
            
    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        向量化GAE计算
        
        Args:
            last_value: (num_envs, 1) 最后状态的价值估计
        """
        size = self.buffer_size if self.full else self.ptr
        
        # Expand last_value for all agents
        last_value_expanded = last_value.unsqueeze(2).expand(-1, -1, self.num_agents)
        
        # Reverse iteration for GAE
        gae = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value_expanded.squeeze(1)
                next_non_terminal = (~self.dones[t]).float().unsqueeze(1)
            else:
                next_value = self.values[t + 1].expand(-1, self.num_agents)
                next_non_terminal = (~self.dones[t]).float().unsqueeze(1)
            
            current_value = self.values[t].expand(-1, self.num_agents)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - current_value
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            self.advantages[t] = gae
            self.returns[t] = gae + current_value
            
    def get_batches(self, batch_size: int):
 
        size = self.buffer_size if self.full else self.ptr
        total_samples = size * self.num_envs * self.num_agents
        
        # Flatten all data
        obs_flat = self.observations[:size].reshape(-1, self.observations.shape[-1])
        actions_flat = self.actions[:size].reshape(-1, self.actions.shape[-1])
        log_probs_flat = self.log_probs[:size].reshape(-1, 1)
        advantages_flat = self.advantages[:size].reshape(-1)
        returns_flat = self.returns[:size].reshape(-1)
        
        # Repeat global states for each agent
        global_states_flat = self.global_states[:size].unsqueeze(2).expand(
            -1, -1, self.num_agents, -1
        ).reshape(-1, self.global_states.shape[-1])
        
        # Shuffle indices
        indices = torch.randperm(total_samples, device=self.device)
        
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_idx = indices[start:end]
            
            yield {
                'observations': obs_flat[batch_idx],
                'global_states': global_states_flat[batch_idx],
                'actions': actions_flat[batch_idx],
                'old_log_probs': log_probs_flat[batch_idx],
                'advantages': advantages_flat[batch_idx],
                'returns': returns_flat[batch_idx],
            }
            
    def reset(self):
        """重置buffer"""
        self.ptr = 0
        self.full = False


class MAPPO_GPU:

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        num_agents: int,
        num_envs: int = 8,
        hidden_dims: List[int] = [256, 256],
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_attention: bool = True,
        attention_heads: int = 4,
        attention_dim: int = 64,
        use_mixed_precision: bool = True,
        device: torch.device = None,
    ):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.num_agents = num_agents
        self.num_envs = num_envs
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        
        # Networks
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        if use_attention:
            self.critic = CriticNetworkWithAttention(
                global_state_dim=global_state_dim,
                num_agents=num_agents,
                hidden_dims=hidden_dims,
                attention_heads=attention_heads,
                attention_dim=attention_dim,
            ).to(self.device)
        else:
            self.critic = self._build_simple_critic(global_state_dim, hidden_dims)
            
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Compile models (PyTorch 2.0+)
        self._try_compile()
        
    def _build_simple_critic(self, global_state_dim: int, hidden_dims: List[int]) -> nn.Module:
        """构建简单Critic网络"""
        layers = []
        prev_dim = global_state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers).to(self.device)
    
    def _try_compile(self):
        
        try:
            # torch.compile 在某些环境下有兼容性问题，禁用以确保稳定运行
            # if hasattr(torch, 'compile'):
            #     self.actor = torch.compile(self.actor, mode='reduce-overhead')
            #     self.critic = torch.compile(self.critic, mode='reduce-overhead')
            #     print("Models compiled with torch.compile")
            pass
        except Exception as e:
            print(f"torch.compile not available: {e}")
            
    @torch.no_grad()
    def get_action(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        # Flatten for actor
        B, N, D = obs.shape
        obs_flat = obs.reshape(B * N, D)
        
        # Actor forward
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with amp.autocast(device_type=device_type, enabled=self.use_mixed_precision):
            actions_flat, log_probs_flat, _ = self.actor.get_action(obs_flat, deterministic)
            values = self.critic(global_state)
        
        # Reshape
        actions = actions_flat.reshape(B, N, -1)
        log_probs = log_probs_flat.reshape(B, N, 1)
        
        return actions, log_probs, values
    
    def update(
        self,
        buffer: GPURolloutBuffer,
        batch_size: int = 256,
        num_epochs: int = 10,
    ) -> Dict[str, float]:
    
    
        
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'approx_kl': 0,
        }
        n_updates = 0
        
        for epoch in range(num_epochs):
            for batch in buffer.get_batches(batch_size):
                obs = batch['observations']
                global_states = batch['global_states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Forward pass with mixed precision
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                with amp.autocast(device_type=device_type, enabled=self.use_mixed_precision):
                    # Actor evaluation
                    new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                    
                    # Critic evaluation
                    values = self.critic(global_states).squeeze(-1)
                    
                    # Policy loss (PPO-clip)
                    ratio = (new_log_probs.squeeze(-1) - old_log_probs.squeeze(-1)).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(values, returns)
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    loss = (
                        policy_loss 
                        + self.value_loss_coef * value_loss 
                        + self.entropy_coef * entropy_loss
                    )
                
                # Backward pass
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    approx_kl = (old_log_probs.squeeze(-1) - new_log_probs.squeeze(-1)).mean()
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += -entropy_loss.item()
                metrics['approx_kl'] += approx_kl.item()
                n_updates += 1
        
        # Average metrics
        for k in metrics:
            metrics[k] /= max(n_updates, 1)
            
        return metrics
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
