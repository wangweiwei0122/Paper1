

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import torch



SCENARIOS: Dict[str, Dict] = {
    # ========== A类场景：CAQ + TGC市场 ==========
    # 基准场景
    "A": {"fine_per_unit": 600.0, "reward_per_unit": 0.0, "include_caq_market": True, "include_tgc_market": True, "description": "Base scenario (CAQ + TGC markets)"},
    
    # 罚款敏感性分析
    "A1": {"fine_per_unit": 300.0, "reward_per_unit": 0.0, "include_caq_market": True, "include_tgc_market": True, "description": "Low fine (CAQ + TGC)"},
    "A2": {"fine_per_unit": 900.0, "reward_per_unit": 0.0, "include_caq_market": True, "include_tgc_market": True, "description": "High fine (CAQ + TGC)"},
    
    # 奖励敏感性分析
    "A3": {"fine_per_unit": 600.0, "reward_per_unit": 20.0, "include_caq_market": True, "include_tgc_market": True, "description": "Low reward (CAQ + TGC)"},
    "A4": {"fine_per_unit": 600.0, "reward_per_unit": 70.0, "include_caq_market": True, "include_tgc_market": True, "description": "Medium reward (CAQ + TGC)"},
    "A5": {"fine_per_unit": 600.0, "reward_per_unit": 200.0, "include_caq_market": True, "include_tgc_market": True, "description": "High reward (CAQ + TGC)"},
    "A6": {"fine_per_unit": 600.0, "reward_per_unit": 300.0, "include_caq_market": True, "include_tgc_market": True, "description": "Very high reward (CAQ + TGC)"},
    "A7": {"fine_per_unit": 600.0, "reward_per_unit": 600.0, "include_caq_market": True, "include_tgc_market": True, "description": "Equal fine-reward (CAQ + TGC)"},
    
    # ========== B类场景：仅CAQ市场（无TGC） ==========
    "B": {"fine_per_unit": 600.0, "reward_per_unit": 0.0, "include_caq_market": True, "include_tgc_market": False, "description": "No TGC market - CAQ only"},
    
    # ========== C类场景：仅TGC市场（无CAQ） ==========
    "C": {"fine_per_unit": 600.0, "reward_per_unit": 0.0, "include_caq_market": False, "include_tgc_market": True, "description": "No CAQ market - TGC only"},
}


def get_scenario_config(scenario_name: str) -> Dict:
    """获取场景配置"""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_name]


@dataclass
class PolicyConfig:
    """RPS政策参数配置"""
    initial_rps_quota: float = 0.15
    rps_growth_step: float = 0.01
    max_rps_quota: float = 0.30
    fine_per_unit: float = 600.0
    reward_per_unit: float = 0.0
    

@dataclass
class MarketConfig:
    """市场参数配置 (论文 Table 1 Section 3.1)"""
    # 论文电价参数 (元/MWh)
    electricity_price_re: float = 406.0      # P^RE_t 可再生能源上网电价
    electricity_price_coal: float = 371.0    # P^TE_t 煤电(火电)上网电价
    electricity_price_retail: float = 590.0  # P^sub_t 终端用户零售电价
    
    # 市场初始价格
    caq_initial_price: float = 70.0
    tgc_initial_price: float = 70.0
    price_range: Tuple[float, float] = (20.0, 200.0)
    max_price_change_rate: float = 0.1
    
    # TGC供应参数 (论文 Section 3.1)
    tgc_initial_supply: float = 7.276e8     # 7276×10⁵ MWh = 7.276×10⁸ MWh
    tgc_supply_growth_rate: float = 0.025   # 2.5% 年增长率


@dataclass
class AgentConfig:
    """智能体参数配置"""
    # Pareto分布参数（Lomax形式）：demand_i = np.random.pareto(a) * m
    # 期望值：E[demand] = m / (a - 1) ≈ 5.8239×10⁷ MWh
    # N=100时初始总需求：≈ 5.8239×10⁹ MWh = 58233×10⁵ MWh (Table1)
    demand_pareto_alpha: float = 10000.0  # shape参数（α）
    demand_pareto_scale: float = 5.8233e11  # scale参数（m，单位MWh）
    re_share_init_range: Tuple[float, float] = (0.14, 0.16)
    memory_length_range: Tuple[int, int] = (3, 7)
    # 随机种子，用于可复现的智能体初始值采样
    seed: int = 42


@dataclass
class EnvironmentConfig:
    """环境配置"""
    num_agents: int = 100
    max_periods: int = 11  # 默认 11 期 -> 2020..2030（含）
    base_year: int = 2020  # 仿真基年：period 0 对应的日历年
    include_caq_market: bool = True
    include_tgc_market: bool = True


@dataclass
class TrainingConfig:
    """训练配置"""
    # Basic
    total_timesteps: int = 100000
    rollout_length: int = 2048
    
    # PPO Hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    batch_size: int = 256
    num_epochs: int = 10
    
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    use_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 64


@dataclass
class GPUConfig:
    """GPU优化配置"""
    device: str = "auto"  # "auto", "cuda", "cpu"
    use_mixed_precision: bool = True
    num_parallel_envs: int = 8  # 并行环境数量
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    pin_memory: bool = True
    num_workers: int = 4
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class LoggingConfig:
    """日志配置"""
    log_dir: str = "results/rps_marl_v4"
    log_interval: int = 1000
    save_interval: int = 10000
    eval_interval: int = 5000
    verbose: int = 1


@dataclass 
class RPSConfig:
    """完整配置"""
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'policy': self.policy.__dict__,
            'market': self.market.__dict__,
            'agent': self.agent.__dict__,
            'environment': self.environment.__dict__,
            'training': {k: v for k, v in self.training.__dict__.items()},
            'gpu': self.gpu.__dict__,
            'logging': self.logging.__dict__,
        }
