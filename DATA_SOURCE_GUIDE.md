# 数据来源说明

## Q3：收敛图（奖励曲线）的数据来自哪里？

### 1. 训练收敛图（train_gpu.py 产生）
**文件**：`train_gpu.py` 第 165-230 行

**数据流**：
```python
# 每个 episode 收集数据
for episode in range(self.num_episodes):
    episode_reward = torch.zeros(self.num_parallel_envs, device=self.device)
    
    for step in range(config.environment.max_periods):  # 11 期
        # 执行环境
        rewards = env.step(actions)[2]
        episode_reward += rewards.sum(dim=1)  # 累积
    
    # 记录每个 episode 的平均奖励
    avg_reward = episode_reward.mean().item()
    episode_rewards.append(avg_reward)  # 关键：存储到列表
```

**输出**：
- 保存为 JSON：`*_rewards.json` （包含 episode_rewards 列表）
- 打印到控制台：`Episode xxx: Reward=0.23`

**使用**：`plot_reward_from_checkpoint.py` 读取 checkpoint 后重新生成仿真数据

---

### 2. 仿真阶段的收敛图（plot_reward_from_checkpoint.py 产生）

**文件**：`plot_reward_from_checkpoint.py` 第 100-150 行

**数据流**：
```python
# 加载已训练的模型
agent.load(checkpoint_path)

# 运行一次完整仿真（11期）
for step_idx in range(config.environment.max_periods):
    actions = agent.get_action(obs, global_state, deterministic=True)
    rewards, infos = env.step(actions)[2:4]
    
    # 收集每期的数据
    avg_reward = rewards[0].sum().item()
    rewards_per_period.append(avg_reward)  # 关键
    compliance_rates.append(infos['compliance_rate'][0])
    re_shares.append(infos['mean_re_share'][0])
```

**输出**：
- PNG 图表：`results/A_episode_0100_reward.png`
- CSV 数据：`results/A_episode_0100_metrics.csv` (包含 reward 列)

**区别**：
- 训练图：100 个 episode 的奖励变化
- 仿真图：1 个 episode（11 期）内每期的奖励变化

---

## Q4：仿真使用的数据从哪里来？

### 1. 市场仿真数据（simulate_gpu_A.py 产生）

**文件**：`simulate_gpu_A.py` / `simulate_gpu_B.py` / `simulate_gpu_C.py`

**数据流**：
```python
# 加载训练好的模型
agent.load(checkpoint_path)

# 循环执行 11 期仿真
for step_idx in range(11):
    # 获取确定性动作（非随机）
    actions = agent.get_action(obs, global_state, deterministic=True)
    
    # 环境步进，收集数据
    next_obs, next_global_state, rewards, dones, infos = env.step(actions)
    
    # 提取市场数据
    year = infos['year'][0]
    compliance_rate = infos['compliance_rate'][0]  # 合规率
    caq_price = infos['caq_price'][0]              # CAQ 价格
    caq_volume = infos['caq_volume'][0]            # CAQ 成交量
    profit_mean = infos['profit_mean'][0]          # 平均利润
    
    # 保存到 DataFrame
    step_data.append({
        'year': year,
        'compliance_rate': compliance_rate,
        'profit_mean': profit_mean,
        'caq_price': caq_price,
        'caq_volume': caq_volume,
        ...
    })
```

**输出**：
- `A_market.csv` - 市场层面数据（每行 1 期）
- `A_agents.csv` - 企业层面数据（每行 1 个企业 × 1 期）

**数据来源链**：
```
model checkpoint → agent.get_action() → env.step() → infos dict → DataFrame → CSV
```

---

### 2. 数据内容详解

**market.csv 列**：
| 列名 | 来自 | 说明 |
|------|------|------|
| year | `infos['year']` | 日历年 (2020-2030) |
| compliance_rate | `infos['compliance_rate']` | 合规企业占比 |
| mean_re_share | `infos['mean_re_share']` | 平均绿电占比 |
| caq_price | `infos['caq_price']` | CAQ 清算价格 |
| caq_volume | `infos['caq_volume']` | CAQ 成交量 |
| profit_mean | `infos['profit_mean']` | 企业平均利润 |
| fines_paid | `infos['fines_paid']` | 总罚款额 |

**agents.csv 列**：
| 列名 | 来自 | 说明 |
|------|------|------|
| year | 循环变量 | 日历年 |
| agent_id | 循环变量 | 企业编号 (0-99) |
| demand | `env.agent_manager.demand[0]` | 企业用电需求 |
| re_share | `env.agent_re_share[0]` | 企业绿电占比 |
| quota_balance | `env.agent_quota_balance[0]` | 企业配额余额 |
| compliance | 计算得出 | 是否合规 (0/1) |
| profit | `env.last_profit_per_agent[0]` | 企业利润 |

---

### 3. 数据获取顺序

```
【初始化阶段】
env.reset() → obs, global_state

【循环阶段】（11 次迭代）
├─ 动作决策：agent.get_action(obs, global_state)
├─ 环境步进：env.step(actions)
│  ├─ 物理计算：绿电消费、配额计算
│  ├─ 市场撮合：CAQ/TGC 交易
│  ├─ 罚款计算：penalty
│  └─ 信息收集：infos
├─ 数据提取：from infos['compliance_rate'], infos['caq_price'], ...
└─ 存储：append to list/DataFrame

【输出阶段】
DataFrame.to_csv() → A_market.csv, A_agents.csv
```

---

## 总结：三种数据来源

| 数据类型 | 来源 | 文件 | 用途 |
|---------|------|------|------|
| **训练收敛** | train_gpu.py | `*_rewards.json` | 监控训练进度 |
| **仿真曲线** | plot_reward_from_checkpoint.py | PNG 图表 | 可视化单个 episode |
| **市场仿真** | simulate_gpu_A/B/C.py | CSV 文件 | 详细数据分析 |

所有数据最终来自：`env.step()` → `infos` dict + 环境内部状态
