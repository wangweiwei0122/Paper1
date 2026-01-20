import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass, field


@dataclass
class RewardContext:
    
    # 电力消费数据
    electricity_consumed: torch.Tensor  # (B, N) 总消电量
    re_consumed: torch.Tensor           # (B, N) 绿电消费量
    non_re_consumed: torch.Tensor       # (B, N) 火电消费量
    
    # 市场交易数据
    caq_trades: torch.Tensor            # (B, N) CAQ市场成交量(+买入,-卖出)
    tgc_trades: torch.Tensor            # (B, N) TGC市场成交量(+买入,-卖出)
    
    # 市场价格
    caq_price: torch.Tensor             # (B,) CAQ清算价格
    tgc_price: torch.Tensor             # (B,) TGC清算价格
    
    # 合规性数据
    quota_balance: torch.Tensor         # (B, N) 配额缺盈（+盈余,-缺口）
    
    # 价格参数
    price_retail: float                 # 零售电价
    price_re: float                     # 绿电上网电价
    price_coal: float                   # 火电上网电价
    
    # 惩奖参数
    fine_per_unit: float                # 缺额罚款单价
    reward_per_unit: float              # 盈余奖励单价

class RewardRegistry:
    """
    奖励函数的动态注册中心，支持灵活添加新的奖励组件
    """
    _rewards: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器：注册新的奖励函数"""
        def decorator(func: Callable):
            cls._rewards[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """根据名称获取奖励函数"""
        return cls._rewards[name]

@RewardRegistry.register("profit")
def profit_reward(env, ctx: RewardContext) -> torch.Tensor:
    """
    利润奖励（主要奖励组件） - 带归一化
    公式：利润 = 售电收入 - 发电成本 - 碳交易支出 - 违规罚款 + 盈余奖励
    归一化：除以平均电力需求量，使奖励大小更稳定
    """
    # 售电收入 = 消电量 × 零售电价
    revenue = ctx.electricity_consumed * ctx.price_retail
    
    # 发电成本 = 火电消费量 × 火电电价 + 绿电消费量 × 绿电电价
    cost = ctx.non_re_consumed * ctx.price_coal + ctx.re_consumed * ctx.price_re
    
    # 碳配额支出 = 成交量 × 价格（卖出为负数，自动转为收入）
    caq_cost = ctx.caq_trades * ctx.caq_price.unsqueeze(1)
    tgc_cost = ctx.tgc_trades * ctx.tgc_price.unsqueeze(1)
    
    # 违规惩罚 = max(0, -余额) × 罚款单价
    penalty = F.relu(-ctx.quota_balance) * ctx.fine_per_unit
    
    # 盈余奖励 = max(0, 余额) × 奖励单价
    bonus = F.relu(ctx.quota_balance) * ctx.reward_per_unit
    
    # 汇总
    profit = revenue - cost - caq_cost - tgc_cost - penalty + bonus
    
    # 【归一化】：除以平均需求量，使奖励幅度更稳定
    # 避免由于需求量变大而导致奖励量无限增长
    avg_demand = ctx.electricity_consumed.mean(dim=1, keepdim=True).clamp(min=1.0)
    profit_normalized = profit / avg_demand
    
    return profit_normalized

@RewardRegistry.register("compliance")
def compliance_reward(env, ctx: RewardContext) -> torch.Tensor:
    """
    合规性奖励（辅助奖励，可选）
    
    1. 实际绿电消费 (RE consumption)
    2. 购买 TGC 交易 (TGC transaction)
    3. 进行 CAQ 交易 (CAQ transaction)
    
    
    """
    # 基于交易后的余额判定：如果有剩余或持平配额，则合规
    compliant = (ctx.quota_balance >= 0).float()
    return compliant





class VectorizedRPSEnv:
    
    INITIAL_RPS_QUOTA = 0.15        # 15% RPS目标
    RPS_GROWTH_STEP = 0.01          # 每期增长1%
    MAX_RPS_QUOTA = 0.30            # 上限30%
    CAQ_INITIAL_PRICE = 70.0
    TGC_INITIAL_PRICE = 70.0
    PRICE_MIN = 20.0
    PRICE_MAX = 200.0
    MAX_PRICE_CHANGE_RATE = 0.1
    ELECTRICITY_PRICE_RE = 406.0        # 绿电电价
    ELECTRICITY_PRICE_COAL = 371.0      # 火电电价
    ELECTRICITY_PRICE_RETAIL = 590.0    # 零售电价
    TGC_INITIAL_SUPPLY = 7.276e8
    TGC_SUPPLY_GROWTH_RATE = 0.025
    OBS_DIM = 9         # 智能体观测维度
    ACTION_DIM = 1      # 智能体动作维度（仅 RE 调整动作）
    GLOBAL_STATE_DIM = 17  # 全局状态维度
    
    def __init__(
        self,
        fine_per_unit: float = 600.0,
        reward_per_unit: float = 0.0,
        include_caq_market: bool = True,
        include_tgc_market: bool = True,
        agent_manager=None,
        device: torch.device = None,
        num_envs: int = 8,
        num_agents: int = 100,
        max_periods: int = 10,
        base_year: int = 2020,
    ):
        
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.B = num_envs  # 环境数量
        self.N = num_agents  # 智能体数量
        self.max_periods = max_periods
        self.base_year = base_year
        self.include_caq_market = include_caq_market
        self.include_tgc_market = include_tgc_market
        self.agent_manager = agent_manager
        
        # 策略参数（使用类常量）
        self.initial_rps_quota = self.INITIAL_RPS_QUOTA
        self.rps_growth_step = self.RPS_GROWTH_STEP
        self.max_rps_quota = self.MAX_RPS_QUOTA
        
        # 奖励参数
        self.fine_per_unit = fine_per_unit
        self.reward_per_unit = reward_per_unit
        
        # 市场参数（使用类常量）
        self.caq_initial_price = self.CAQ_INITIAL_PRICE
        self.tgc_initial_price = self.TGC_INITIAL_PRICE
        self.price_min = self.PRICE_MIN
        self.price_max = self.PRICE_MAX
        self.max_price_change_rate = self.MAX_PRICE_CHANGE_RATE
        
        # 电价参数（使用类常量）
        self.electricity_price_re = self.ELECTRICITY_PRICE_RE
        self.electricity_price_coal = self.ELECTRICITY_PRICE_COAL
        self.electricity_price_retail = self.ELECTRICITY_PRICE_RETAIL
        
        # TGC供应参数（使用类常量）
        self.tgc_initial_supply = self.TGC_INITIAL_SUPPLY
        self.tgc_supply_growth_rate = self.TGC_SUPPLY_GROWTH_RATE
        
        # 观测维度（使用类常量）
        self.obs_dim = self.OBS_DIM
        self.action_dim = self.ACTION_DIM
        self.global_state_dim = self.GLOBAL_STATE_DIM
        
        # 历史记录
        self._price_history_caq: List[torch.Tensor] = []
        self._price_history_tgc: List[torch.Tensor] = []
        
        self.reset()
    
    
    
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.current_period = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.current_rps_quota = torch.full(
            (self.B,), self.initial_rps_quota,
            dtype=torch.float32, device=self.device
        )
        
        
        self.agent_manager.reset()
        
        # 同步状态
        self.agent_demand = self.agent_manager.demand
        self.agent_re_share = self.agent_manager.re_share
        # 每期独立：从agent_manager同步配额为0（新的一期开始时配额为0）
        self.agent_quota_balance = self.agent_manager.quota_balance
        self.agent_epsilon = self.agent_manager.epsilon
        
        # 初始化库存和价格
        self.agent_caq_inventory = torch.zeros(self.B, self.N, device=self.device)
        self.agent_tgc_inventory = torch.zeros(self.B, self.N, device=self.device)
        # 初始化上期成交量（用于观测）
        self.last_caq_trade_volumes = torch.zeros(self.B, self.N, device=self.device)
        self.last_tgc_trade_volumes = torch.zeros(self.B, self.N, device=self.device)
        self.caq_price = torch.full((self.B,), self.caq_initial_price, dtype=torch.float32, device=self.device)
        self.tgc_price = torch.full((self.B,), self.tgc_initial_price, dtype=torch.float32, device=self.device)
        self.done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        
        self._price_history_caq = [self.caq_price.clone()]
        self._price_history_tgc = [self.tgc_price.clone()]
        
        # 初始化周期跟踪
        self._init_period_tracking()

        return self._get_observations(), self._get_global_state()
    
    def register_agent_manager(self, manager):
        manager.num_envs = self.num_envs
        manager.num_agents = self.num_agents
        manager.device = self.device
        self.agent_manager = manager
    
    def _init_period_tracking(self):
        """初始化周期数据跟踪（用于合规率和诊断分析）"""
        # 在每个step开始时保存本周期的关键数据
        self.last_period_electricity_consumed = None
        self.last_period_re_consumed = None
        self.last_period_required_quota = None
        self.last_period_re_share = None
        # 初始化利润跟踪（用于仿真分析）
        self.last_profit_per_agent = torch.zeros(self.B, self.N, device=self.device)
    
    
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        # 【物理1】动作执行：调整RE份额
        # 智能体通过调节绿电比例来应对RPS政策压力
        re_adjust = actions[..., 0]  # 提取第一维动作(RE调整)
        # 限制动作幅度：最多每期调整10%
        re_delta = re_adjust * 0.10
        new_re_share = (self.agent_re_share + re_delta).clamp(0.0, 1.0)
        
        # 【物理2】需求增长：全社会用电量的自然增长
        # 需求增长由智能体的sync_internal_state方法管理
        electricity_consumed = self.agent_demand
        
        # 【物理3】发电结构划分
        # 总消电 = 绿电 + 火电
        re_consumed = electricity_consumed * new_re_share
        non_re_consumed = electricity_consumed * (1 - new_re_share)
        
        # 【物理4】配额计算（核心的合规逻辑）
        # 政策义务 = 总消电 × 当前RPS比例
        required_quota = electricity_consumed * self.current_rps_quota.unsqueeze(1)
        # 绿电证书 = 实际绿电发电量（1度绿电=1个证书）
        earned_certificates = re_consumed
        
        # 【保存周期数据】用于合规率和诊断分析
        self.last_period_electricity_consumed = electricity_consumed.clone()
        self.last_period_re_consumed = re_consumed.clone()
        self.last_period_required_quota = required_quota.clone()
        self.last_period_re_share = new_re_share.clone()
        
        # 更新余额：期初 + 本期产生 - 本期义务
        # 正余额：超额达成，可卖出；负余额：缺额，需买入
        self.agent_quota_balance = (
            self.agent_quota_balance + earned_certificates - required_quota
        )
        
        # 更新RE比例
        self.agent_re_share = new_re_share
        
        # 将最新的物理和财务状态同步给Agent Manager
       
        if self.agent_manager is not None:
            self.agent_manager.sync_internal_state(
                new_re_share,               # 本期RE份额（使用new_re_share）
                self.agent_quota_balance,   # 物理结算后的余额（可能为负）
            )
            # 同步修改后的agent_manager属性（但不覆盖re_share和quota_balance，保持计算值）
            self.agent_demand = self.agent_manager.demand
            # self.agent_re_share = self.agent_manager.re_share  # 不覆盖，用计算值
            # self.agent_quota_balance保持之前计算的值

        
        
        caq_clearing_price = self.caq_price.clone()
        caq_trade_volumes = torch.zeros(self.B, self.N, device=self.device)
        
        if self.include_caq_market:
            caq_clearing_price, caq_trade_volumes = self._run_caq_market()
            self.caq_price = caq_clearing_price
            self.agent_caq_inventory += caq_trade_volumes
        
       
        tgc_clearing_price = self.tgc_price.clone()
        tgc_trade_volumes = torch.zeros(self.B, self.N, device=self.device)
        
        if self.include_tgc_market:
            tgc_clearing_price, tgc_trade_volumes = self._run_tgc_market()
            self.tgc_price = tgc_clearing_price
            self.agent_tgc_inventory += tgc_trade_volumes
        
        
        self.last_caq_trade_volumes = self.agent_caq_inventory.clone()
        self.last_tgc_trade_volumes = self.agent_tgc_inventory.clone()
        
        # 市场成交的配额加入账户余额
        self.agent_quota_balance += (
            self.agent_caq_inventory + self.agent_tgc_inventory
        )
        
        self.agent_caq_inventory.zero_()
        self.agent_tgc_inventory.zero_()
        
       
        
        reward_ctx = RewardContext(
            electricity_consumed=electricity_consumed,
            re_consumed=re_consumed,
            non_re_consumed=non_re_consumed,
            caq_trades=caq_trade_volumes,
            tgc_trades=tgc_trade_volumes,
            caq_price=caq_clearing_price,
            tgc_price=tgc_clearing_price,
            quota_balance=self.agent_quota_balance,
            price_retail=self.electricity_price_retail,
            price_re=self.electricity_price_re,
            price_coal=self.electricity_price_coal,
            fine_per_unit=self.fine_per_unit,
            reward_per_unit=self.reward_per_unit,
        )
        
        # 【计算未归一化的利润】用于仿真分析
        # 这个是原始的利润数据（未经过归一化），用于CSV输出和分析
        revenue = reward_ctx.electricity_consumed * reward_ctx.price_retail
        cost = reward_ctx.non_re_consumed * reward_ctx.price_coal + reward_ctx.re_consumed * reward_ctx.price_re
        caq_cost = reward_ctx.caq_trades * reward_ctx.caq_price.unsqueeze(1)
        tgc_cost = reward_ctx.tgc_trades * reward_ctx.tgc_price.unsqueeze(1)
        penalty = F.relu(-reward_ctx.quota_balance) * reward_ctx.fine_per_unit
        bonus = F.relu(reward_ctx.quota_balance) * reward_ctx.reward_per_unit
        # 【关键】未归一化的利润 - 用于仿真分析
        self.last_profit_per_agent = revenue - cost - caq_cost - tgc_cost - penalty + bonus
        
        rewards = self._calculate_rewards(reward_ctx)
        
        # 在罚款之前收集 infos，使 compliance_rate 基于交易后、罚款前的余额
        observations = self._get_observations()
        global_state = self._get_global_state()
        infos = self._collect_info(
            caq_clearing_price, tgc_clearing_price,
            caq_trade_volumes, tgc_trade_volumes,
            fines_paid=None  # 暂时不传，计算后再传
        )
        
        # 【惩罚阶段】计算罚款（此时合规率已基于交易后余额计算）
        fines_paid = F.relu(-self.agent_quota_balance) * self.fine_per_unit
        # 更新 infos 中的 fines_paid
        infos['fines_paid'] = fines_paid.sum(dim=1)
        
        # 清零余额以准备下一个周期
        self.agent_quota_balance = torch.zeros_like(self.agent_quota_balance)
        
        
        self.current_period = self.current_period + 1
        
        # RPS配额
       
        self.current_rps_quota = (
            self.current_rps_quota + self.rps_growth_step
        ).clamp(max=self.max_rps_quota)
        
       
        self.done = self.current_period >= self.max_periods
        
        
        self._price_history_caq.append(self.caq_price.clone())
        self._price_history_tgc.append(self.tgc_price.clone())
        
        
        # Agent Manager根据价格变化调整对下一期的预期
        if self.agent_manager is not None:
            self.agent_manager.update_price_forecast(
                self.get_price_history('caq'), market_type='caq'
            )
            self.agent_manager.update_price_forecast(
                self.get_price_history('tgc'), market_type='tgc'
            )
        
        return observations, global_state, rewards, self.done, infos
    

    
    def _get_observations(self) -> torch.Tensor:  #定义actor的输入
        obs = torch.stack([
            
            torch.log1p(self.agent_demand) / 20.0,
            self.agent_re_share,

            self.agent_quota_balance / 1e6,
           
            self.current_rps_quota.unsqueeze(1).expand(self.B, self.N),
          
            (self.current_period.float() / self.max_periods).unsqueeze(1).expand(self.B, self.N),
            
            self.caq_price.unsqueeze(1).expand(self.B, self.N) / self.price_max,
   
          
            self.tgc_price.unsqueeze(1).expand(self.B, self.N) / self.price_max,
           
           
            self.last_caq_trade_volumes / 1e6,
            
            self.last_tgc_trade_volumes / 1e6,
        ], dim=-1)
        
        return obs
    
    def _get_global_state(self) -> torch.Tensor:
        global_features = [
            # 宏观1：平均需求，反映市场规模
            self.agent_demand.mean(dim=1) / 1e8,
            # 宏观2：需求差异，反映市场异质性
            self.agent_demand.std(dim=1) / 1e8,
            # 宏观3：平均绿电程度
            self.agent_re_share.mean(dim=1),
            # 宏观4：绿电差异，反映企业差异
            self.agent_re_share.std(dim=1),
            # 宏观5：平均配额状态
            self.agent_quota_balance.mean(dim=1) / 1e6,
            # 宏观6：CAQ总库存(买卖方向)
            self.agent_caq_inventory.sum(dim=1) / 1e6,
            # 宏观7：TGC总库存
            self.agent_tgc_inventory.sum(dim=1) / 1e6,
            # 宏观8：CAQ市场价格
            self.caq_price / self.price_max,
            # 宏观9：TGC市场价格
            self.tgc_price / self.price_max,
            # 宏观10：时间进度
            self.current_period.float() / self.max_periods,
            # 宏观11：政策强度
            self.current_rps_quota,
        ]
        
        # 确保长度一致
        while len(global_features) < self.global_state_dim:
            global_features.append(torch.zeros(self.B, device=self.device))
        
        return torch.stack(global_features[:self.global_state_dim], dim=-1)
    
    
    def _run_caq_market(self) -> Tuple[torch.Tensor, torch.Tensor]:
        clearing_prices = self.caq_price.clone()
        trade_volumes = torch.zeros(self.B, self.N, device=self.device)
        all_bids = None
        if self.agent_manager is not None:
            is_buyer_mask = self.agent_quota_balance < 0
            ref_price_expanded = self.caq_price.unsqueeze(1).expand(self.B, self.N)
            all_bids = self.agent_manager.get_bid_price(
                'caq', ref_price_expanded, is_buyer_mask
            )
        
        # 逐个环境进行撮合
        for env_idx in range(self.B):
            quota = self.agent_quota_balance[env_idx]
            
            # 识别买卖方
            buyers = (quota < 0).nonzero(as_tuple=True)[0]
            sellers = (quota > 0).nonzero(as_tuple=True)[0]
            
            # 如果某一方为空，市场无法交易
            if len(buyers) == 0 or len(sellers) == 0:
                continue
            
            # 提取报价（必须从Agent Manager提供）
            assert all_bids is not None, "CAQ市场必须通过Agent Manager提供报价"
            buy_prices = all_bids[env_idx, buyers]
            sell_prices = all_bids[env_idx, sellers]
            
            # 交易手数：买方缺额、卖方盈余
            buy_quantities = -quota[buyers]
            sell_quantities = quota[sellers]
            
            # 价格排序：高价买方优先，低价卖方优先
            buy_sorted_idx = torch.argsort(buy_prices, descending=True)
            sell_sorted_idx = torch.argsort(sell_prices, descending=False)
            
            sorted_buy_prices = buy_prices[buy_sorted_idx]
            sorted_sell_prices = sell_prices[sell_sorted_idx]
            sorted_buy_qtys = buy_quantities[buy_sorted_idx].clone()
            sorted_sell_qtys = sell_quantities[sell_sorted_idx].clone()
            
            sorted_buyers = buyers[buy_sorted_idx]
            sorted_sellers = sellers[sell_sorted_idx]
            
            # 按价格优先逐笔匹配
            buy_ptr, sell_ptr = 0, 0
            last_buy_price, last_sell_price = (
                sorted_buy_prices[0], sorted_sell_prices[0]
            )
            total_matched = 0.0
            
            while buy_ptr < len(sorted_buyers) and sell_ptr < len(sorted_sellers):
                # 价格交叉检查：买价必须≥卖价才能成交
                if sorted_buy_prices[buy_ptr] < sorted_sell_prices[sell_ptr]:
                    break
                
                # 成交量：两者最小值
                match_qty = min(
                    sorted_buy_qtys[buy_ptr].item(),
                    sorted_sell_qtys[sell_ptr].item()
                )
                
                if match_qty > 0:
                    buyer_id = sorted_buyers[buy_ptr]
                    seller_id = sorted_sellers[sell_ptr]
                    
                    # 记录成交：买方为正，卖方为负
                    trade_volumes[env_idx, buyer_id] += match_qty
                    trade_volumes[env_idx, seller_id] -= match_qty
                    
                    # 扣除已成交手数
                    sorted_buy_qtys[buy_ptr] -= match_qty
                    sorted_sell_qtys[sell_ptr] -= match_qty
                    total_matched += match_qty
                    
                    # 更新最后成交价
                    last_buy_price = sorted_buy_prices[buy_ptr]
                    last_sell_price = sorted_sell_prices[sell_ptr]
                
                # 移动指针：委托手数用尽则转向下一个
                if sorted_buy_qtys[buy_ptr] <= 1e-6:
                    buy_ptr += 1
                if sorted_sell_qtys[sell_ptr] <= 1e-6:
                    sell_ptr += 1
            
            # 成交价：最后买卖价的均值
            if total_matched > 0:
                clearing_prices[env_idx] = 0.5 * (last_buy_price + last_sell_price)
        
        return clearing_prices.clamp(self.price_min, self.price_max), trade_volumes
    
    def _run_tgc_market(self) -> Tuple[torch.Tensor, torch.Tensor]:
        
        clearing_prices = self.tgc_price.clone()
        trade_volumes = torch.zeros(self.B, self.N, device=self.device)
        
        # 从Agent Manager获取报价
        all_bids = None
        if self.agent_manager is not None:
            is_buyer_mask = self.agent_quota_balance < 0
            ref_price_expanded = self.tgc_price.unsqueeze(1).expand(self.B, self.N)
            all_bids = self.agent_manager.get_bid_price(
                'tgc', ref_price_expanded, is_buyer_mask
            )
        
        # 逐个环境进行撮合
        for env_idx in range(self.B):
            quota = self.agent_quota_balance[env_idx]
            
            # TGC只有买方（缺配额的企业）
            buyers = (quota < 0).nonzero(as_tuple=True)[0]
            if len(buyers) == 0:
                continue
            
            
            # 政府供应 = 初始供应 × (1 + 年增长率)^period
            # period 单位为年，增长率为 2.5% 年增长
            period = self.current_period[env_idx].item()
            tgc_supply = self.tgc_initial_supply * (
                (1.0 + self.tgc_supply_growth_rate) ** period
            )
            
            # 买方报价（必须从Agent Manager提供）
            assert all_bids is not None, "TGC市场必须通过Agent Manager提供报价"
            buy_prices = all_bids[env_idx, buyers]
            
            # 买方需求
            buy_quantities = -quota[buyers]
            
            # 排序：按报价从高到低
            sorted_idx = torch.argsort(buy_prices, descending=True)
            sorted_prices = buy_prices[sorted_idx]
            sorted_qtys = buy_quantities[sorted_idx]
            sorted_buyers = buyers[sorted_idx]
            
            remaining_supply = float(tgc_supply)
            last_price = sorted_prices[0]
            
            # 依次分配供应给高价者
            for buyer_id, qty, price in zip(sorted_buyers, sorted_qtys, sorted_prices):
                if remaining_supply <= 0:
                    break
                
                allocated = min(qty.item(), remaining_supply)
                if allocated > 0:
                    trade_volumes[env_idx, buyer_id] = allocated
                    remaining_supply -= allocated
                    last_price = price
            
            # 清算价：最后成交价
            if remaining_supply < float(tgc_supply):
                clearing_prices[env_idx] = last_price
        
        return clearing_prices.clamp(self.price_min, self.price_max), trade_volumes
    

    
    def _calculate_rewards(self, ctx: RewardContext) -> torch.Tensor:
        
        # 售电收入
        revenue = ctx.electricity_consumed * ctx.price_retail
        
        # 发电成本
        cost = ctx.non_re_consumed * ctx.price_coal + ctx.re_consumed * ctx.price_re
        
        # 交易成本
        caq_cost = ctx.caq_trades * ctx.caq_price.unsqueeze(1)
        tgc_cost = ctx.tgc_trades * ctx.tgc_price.unsqueeze(1)
        
        # 罚款与奖励
        penalty = F.relu(-ctx.quota_balance) * ctx.fine_per_unit
        bonus = F.relu(ctx.quota_balance) * ctx.reward_per_unit
        
        # 利润 = 收入 - 成本 - 罚款 + 奖励
        profit = revenue - cost - caq_cost - tgc_cost - penalty + bonus
        
        # 归一化（除以平均需求，避免需求量变大导致奖励爆炸）
        avg_demand = ctx.electricity_consumed.mean(dim=1, keepdim=True).clamp(min=1.0)
        profit_normalized = profit / avg_demand
        
       
        reward = profit_normalized * 1e-3
        
        return reward
    
    def _collect_info( #收集信息的
        self,
        caq_price: torch.Tensor,
        tgc_price: torch.Tensor,
        caq_volumes: torch.Tensor,
        tgc_volumes: torch.Tensor,
        fines_paid: torch.Tensor = None
    ) -> Dict:
        
        
       
        year_tensor = self.current_period + self.base_year
        
        # 若未传入fines_paid，则设为全零
        if fines_paid is None:
            fines_paid = torch.zeros(self.B, self.N, device=self.device)
        compliant = (self.agent_quota_balance >= 0).float()
        compliance_rate = compliant.mean(dim=1)  # 每个环境的合规率
        
      
        avg_electricity = self.last_period_electricity_consumed.mean(dim=1)
        avg_required = self.last_period_required_quota.mean(dim=1)
        avg_re_consumed = self.last_period_re_consumed.mean(dim=1)
        avg_deficit = F.relu(avg_required - avg_re_consumed)
        profit_mean = self.last_profit_per_agent.mean(dim=1)  # (B,) 每个环境的平均利润
        profit_std = self.last_profit_per_agent.std(dim=1)  # (B,) 每个环境的利润标准差

        return {
            'year': year_tensor,
            'compliance_rate': compliance_rate,
            'mean_re_share': self.agent_re_share.mean(dim=1),
            'std_re_share': self.agent_re_share.std(dim=1),
            'caq_price': caq_price,
            'tgc_price': tgc_price,
            'caq_volume': caq_volumes.abs().sum(dim=1),
            'tgc_volume': tgc_volumes.sum(dim=1),
            'num_caq_buyers': (self.agent_quota_balance < 0).sum(dim=1),
            'num_caq_sellers': (self.agent_quota_balance > 0).sum(dim=1),
            'num_tgc_buyers': (self.agent_quota_balance < 0).sum(dim=1),
            'avg_obligated_profit': torch.zeros(self.num_envs, device=self.device),
            'total_quota_deficit': F.relu(-self.agent_quota_balance).sum(dim=1),
            'total_quota_surplus': F.relu(self.agent_quota_balance).sum(dim=1),
            'fines_paid': fines_paid.sum(dim=1),  # 环境层总罚款
            'avg_electricity_consumed': avg_electricity,
            'avg_required_quota': avg_required,
            'avg_re_consumed': avg_re_consumed,
            'avg_deficit_before_trading': avg_deficit,
            'profit_mean': profit_mean,  # 原始利润均值（未归一化）
            'profit_std': profit_std,  # 原始利润标准差（未归一化）
        }
    
    def get_price_history(self, market: str = 'caq') -> torch.Tensor:
        """
        返回价格历史序列 (Batch, Time)
        """
        history = (
            self._price_history_caq if market == 'caq'
            else self._price_history_tgc
        )
        if len(history) == 0:
            return torch.empty(self.B, 0, device=self.device)
        return torch.stack(history, dim=1)





