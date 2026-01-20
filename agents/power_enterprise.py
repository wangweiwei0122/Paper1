import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from configs.config import AgentConfig, EnvironmentConfig, MarketConfig, GPUConfig


class PowerEnterpriseAgentV4:

    def __init__(
        self,
        num_envs: Optional[int] = None,
        num_agents: Optional[int] = None,
        agent_config: Optional[AgentConfig] = None,
        env_config: Optional[EnvironmentConfig] = None,
        market_config: Optional[MarketConfig] = None,
        gpu_config: Optional[GPUConfig] = None,
        seed: Optional[int] = None,
    ):
        # 加载配置（使用默认值或传入值）
        if agent_config is None:
            agent_config = AgentConfig()
        if env_config is None:
            env_config = EnvironmentConfig()
        if market_config is None:
            market_config = MarketConfig()
        if gpu_config is None:
            gpu_config = GPUConfig()
        
        # 优先级：显式参数 > 配置对象 > 默认值
        self.num_envs = num_envs or gpu_config.num_parallel_envs
        self.num_agents = num_agents or env_config.num_agents
        self.seed = seed or agent_config.seed
        
        self.device = gpu_config.get_device()
        self.initial_price = market_config.caq_initial_price
        self.memory_length_range = agent_config.memory_length_range
        self.epsilon_range = (0.0, 0.05)  # 市场竞争强度范围
        # 价格边界：用于限制报价在合理范围内
        self.price_min, self.price_max = market_config.price_range
        # Pareto分布参数（论文Table 1）
        self.demand_pareto_alpha = agent_config.demand_pareto_alpha
        self.demand_pareto_scale = agent_config.demand_pareto_scale
        self.re_share_init_range = agent_config.re_share_init_range
        self.demand_growth_rate = 0.0567  # 年均需求增长率
        # 初始化 generator，用于所有后续采样
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(int(seed))
        ml, mh = self.memory_length_range
        self.fixed_memory_length = torch.randint(ml, mh + 1, (self.num_envs, self.num_agents), generator=self._rng, device=self.device)
        emin, emax = self.epsilon_range
        self.fixed_epsilon = torch.rand(self.num_envs, self.num_agents, generator=self._rng, device=self.device) * (emax - emin) + emin
        B, N = self.num_envs, self.num_agents
        # Pareto分布采样（Lomax形式）：demand_i = np.random.pareto(a) * m
        # 与论文 Simulation code/source/data_loader.py 中的实现保持一致
        # 设定numpy种子以保证采样一致性
        if self.seed is not None:
            np.random.seed(int(self.seed))
        pareto_samples = np.random.pareto(self.demand_pareto_alpha, size=(B, N))
        self.fixed_demand = torch.tensor(pareto_samples * self.demand_pareto_scale, device=self.device, dtype=torch.float32)
        re_low, re_high = self.re_share_init_range
        self.fixed_re_share = (torch.rand(B, N, generator=self._rng, device=self.device) * (re_high - re_low) + re_low)
        
        self.reset()

    def reset(self):# 重置智能体状态
       
        self.demand = self.fixed_demand.clone()
        self.re_share = self.fixed_re_share.clone()
        self.quota_balance = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.price_forecast = torch.full((self.num_envs, self.num_agents), self.initial_price, device=self.device)
        self.tgc_price_forecast = torch.full((self.num_envs, self.num_agents), self.initial_price, device=self.device)
        self.epsilon = self.fixed_epsilon.clone()  
        self.memory_length = self.fixed_memory_length.clone()
    
    def sync_internal_state(
        self,
        re_share: torch.Tensor,
        quota_balance: torch.Tensor,
    ):
        # 同步环境状态：更新当期的RE份额和余额
        # demand增长通过独立的demand_growth_rate在此处应用
        self.demand = self.demand * (1.0 + self.demand_growth_rate)
        self.re_share =re_share
        self.quota_balance=quota_balance
        
    
    
    def update_price_forecast(
        self,
        price_history: torch.Tensor,
        market_type: str = 'caq'
    ):
        """
        基于价格历史和记忆长度更新价格预测。
        - 输入：price_history 形状 (num_envs, T)，其中 T 是历史长度，最新价为 price_history[:, -1]。
        - 每个 agent i 有记忆长度 l_i（self.memory_length），表示仅使用最近 l_i 个期的数据来估计下一期价格。
        设 p_t 为当前期 t 的价格，p_{t-l_i+1} 为窗口起点价格，则平均每期增长率 g_i 计算为：
            g_i = (p_t / p_{t-l_i+1} - 1) / (l_i - 1)
        预测下一期价格 F_{i,t+1} 定义为：
            F_{i,t+1} = p_t * (1 + g_i)
        """
        if price_history.shape[1] < 2:
            return
        
        # 历史长度 T（按 env 批次一致）
        T = price_history.shape[1]
        
        # 当前（最新）价格 p_t：形状 (num_envs,)
        p_t = price_history[:, -1]
        
        # 每个 agent 的记忆长度 l_i（裁剪到 [1, T]）
        l_i = self.memory_length.clone()
        l_i = torch.clamp(l_i, min=1, max=T)
        
        # 计算每个 agent 窗口起点的索引：start = T - l_i
        start_idx = (T - l_i).long()
        ph_exp = price_history.unsqueeze(1).expand(-1, self.num_agents, -1)
        gather_idx = start_idx.unsqueeze(-1)
        # p_prev 为窗口起点的价格（形状 (num_envs, num_agents)）
        p_prev = torch.gather(ph_exp, 2, gather_idx).squeeze(-1)
        
        # p_now 为当前价格在每个 agent 的扩展表示
        p_now = p_t.unsqueeze(1).expand(-1, self.num_agents)

        # 计算平均每期增长率 g_i（加 eps 防止除零），等价于上面的 g_i
        # g_i = (p_now / p_prev - 1) / (l_i - 1)
        eps = 1e-6
        trends = (p_now / (p_prev + eps) - 1.0) / (l_i.float() - 1.0 + eps)

        # 截断增长率以稳定预测（实现层面，论文没有明确给出截断值）
        trends = torch.clamp(trends, -0.2, 0.2)

        # 预测 = 当前价 * (1 + 平均每期增长率)
        forecast = p_now * (1.0 + trends)

        if market_type == 'caq':
            # 更新 CAQ 市场预测
            self.price_forecast = forecast
        else:
            # 更新 TGC 市场预测
            self.tgc_price_forecast = forecast
    
    def get_bid_price(
        self,
        market_type: str,
        ref_price: torch.Tensor,
        is_buyer: torch.Tensor
    ) -> torch.Tensor:
        """
        根据市场类型和买卖身份生成报价
        职责：返回基于agent预期和风险偏好的个性化报价
        
        Args:
            market_type: 'caq' 或 'tgc'
            ref_price: 参考价格，形状 (B, N)，仅在base_price无法获取时使用
            is_buyer: 布尔掩码，形状 (B, N)，True=买方(quota<0), False=非买方(quota>=0)
        
        Returns:
            bids: 报价，形状 (B, N)，已限制在 [price_min, price_max] 范围内
        
        Note:
            - quota==0 的中性agent被视为非买方，不参与CAQ市场交易
            - 环境层在调用时通过额外的quota>0掩码来精确识别卖方
            - 报价已限制到市场允许的价格范围，避免异常出价
        """
        # 选择基准价格：优先使用agent自身预测（理性预期）
        if market_type == 'caq':
            base_price = self.price_forecast
        elif market_type == 'tgc':
            base_price = self.tgc_price_forecast
        else:
            # 兜底：市场类型不认识时使用参考价
            base_price = ref_price
        
        epsilon = self.epsilon
        
        # 初始化报价矩阵
        bids = torch.zeros_like(base_price)
        
        # 买方报价：缺额急需，愿意加价 → Bid = Price × (1 + ε)
        # 非买方（包括卖方和中性）默认为0，环境层会使用quota>0掩码提取真实卖方报价
        is_seller = ~is_buyer
        bids[is_buyer] = base_price[is_buyer] * (1.0 + epsilon[is_buyer])
        
        # 卖方报价：盈余不急，愿意降价吸引 → Bid = Price × (1 - ε)
        bids[is_seller] = base_price[is_seller] * (1.0 - epsilon[is_seller])
        

        bids = bids.clamp(min=self.price_min, max=self.price_max)
        
        return bids

#便于快速生成智能体的，这个就是自然调用智能体的管理部分
def create_agent_manager(
    num_envs: Optional[int] = None,
    num_agents: Optional[int] = None,
    agent_config: Optional[AgentConfig] = None,
    env_config: Optional[EnvironmentConfig] = None,
    market_config: Optional[MarketConfig] = None,
    gpu_config: Optional[GPUConfig] = None,
    seed: Optional[int] = None,
) -> PowerEnterpriseAgentV4:
   
    return PowerEnterpriseAgentV4(
        num_envs=num_envs,
        num_agents=num_agents,
        agent_config=agent_config,
        env_config=env_config,
        market_config=market_config,
        gpu_config=gpu_config,
        seed=seed,
    )
