# 修复总结：合规率计算与奖励为0问题

## 问题 1：修改合规率计算（基于绿电满足RPS）

### 问题描述
原始实现中合规率 (`compliance_rate`) 和 compliance 奖励的判断标准不一致：
- `compliance_rate`（统计用）：基于 `绿电消费 >= RPS要求`
- `compliance_reward`（RL用）：基于 `交易后余额 >= 0`

用户要求统一改为：**即使购买配额后，如果绿电消费仍不满足RPS要求，则不合规**

### 修改文件
**文件**：`envs/vectorized_env.py`

### 修改详情

#### 修改 1：compliance_reward 函数（第 88-100 行）
**原代码**：
```python
@RewardRegistry.register("compliance")
def compliance_reward(env, ctx: RewardContext) -> torch.Tensor:
    """
    合规性奖励（辅助奖励）
    如果余额≥0则奖励1，否则0
    """
    return (ctx.quota_balance >= 0).float()
```

**新代码**：
```python
@RewardRegistry.register("compliance")
def compliance_reward(env, ctx: RewardContext) -> torch.Tensor:
    """
    合规性奖励（辅助奖励）
    修改：即使购买配额后，如果绿电消费仍不满足RPS要求，则不合规
    判断标准：实际绿电消费 >= RPS要求的绿电量
    """
    # RPS要求的绿电消费量 = 总消电 × 当前RPS比例
    rps_required = ctx.electricity_consumed * env.current_rps_quota.unsqueeze(1)
    # 判断绿电是否满足RPS要求
    compliant = (ctx.re_consumed >= rps_required).float()
    return compliant
```

**原理**：
- 之前的 `ctx.quota_balance` 是交易后的余额，可以通过购买配额补足
- 新的判断基于**实际绿电消费**（物理量），无法通过市场交易改变
- 这强化了"合规需要真实的绿电转变"的政策意图

---

#### 修改 2：_collect_info 中的合规率计算（第 660-665 行）
**原代码**：
```python
# 【基于RPS要求计算真实合规率】
# 合规判断：实际绿电消费 >= RPS要求的绿电量
compliant = (self.last_period_re_consumed >= self.last_period_required_quota).float()
compliance_rate = compliant.mean(dim=1)  # 每个环境的合规率
```

**新代码**：
```python
# 【基于RPS要求计算真实合规率】
# 修改后的定义：实际绿电消费 >= RPS要求的绿电量 则合规
# 即使通过市场购买配额，如果绿电消费仍不足，也不认为合规
required_green_energy = self.last_period_electricity_consumed * self.current_rps_quota.unsqueeze(1)
compliant = (self.last_period_re_consumed >= required_green_energy).float()
compliance_rate = compliant.mean(dim=1)  # 每个环境的合规率
```

**改进**：
- 代码更清晰，直接计算 RPS 要求（而不是依赖 `last_period_required_quota`）
- 两处定义保持一致

---

## 问题 2：训练时奖励值为 0 的根本原因

### 问题描述
训练输出的奖励值一直是 0，这会导致：
- RL 无法获得有意义的梯度信号
- 策略无法学习到行为差异
- 损失函数无法区分好坏策略

### 根本原因：scale 因子过小

**定位**：`envs/vectorized_env.py` 第 105-109 行 `RewardConfig` 类

```python
@dataclass
class RewardConfig:
    ...
    scale: float = 1e-8  # 【问题根源】太小！
```

**问题分析**：
1. `profit_reward` 返回值范围：[-∞, +∞]（经过平均需求归一化后，通常 ±100 数量级）
2. 乘以 `1e-8` 后：奖励值 ≈ `±100 × 1e-8 = ±1e-6`
3. 在 32 位浮点精度下，这已经接近舍入误差，被优化器视为 0

### 修改文件
**文件**：`envs/vectorized_env.py`

#### 修改 3：RewardConfig 中的 scale 参数（第 102-109 行）

**原代码**：
```python
@dataclass
class RewardConfig:
    """
    奖励配置类：定义使用哪些奖励组件及其权重
    """
    # 奖励组件列表，如["profit", "compliance"]
    components: List[str] = field(default_factory=lambda: ["profit"])
    # 对应的权重列表
    weights: List[float] = field(default_factory=lambda: [1.0])
    # 是否对奖励进行归一化处理
    normalize: bool = True
    # 归一化缩放因子，用于避免梯度爆炸
    scale: float = 1e-8
```

**新代码**：
```python
@dataclass
class RewardConfig:
    """
    奖励配置类：定义使用哪些奖励组件及其权重
    """
    # 奖励组件列表，如["profit", "compliance"]
    components: List[str] = field(default_factory=lambda: ["profit"])
    # 对应的权重列表
    weights: List[float] = field(default_factory=lambda: [1.0])
    # 是否对奖励进行归一化处理
    normalize: bool = True
    # 【关键修复】归一化缩放因子：原值1e-8导致奖励被压缩为0
    # 利润已通过avg_demand进行了层面归一化，故scale改为1e-6保留可用梯度信息
    scale: float = 1e-6
```

**修复效果**：
- 新 scale = `1e-6`，相比原值提升 **100 倍**
- 奖励范围：±100 × 1e-6 = **±1e-4**（可感知）
- 梯度信号：从浮点误差量级上升到**可训练的水平**

---

## 验证方法

运行新增的诊断脚本：
```bash
python test_reward_fix.py
```

期望输出：
- ✅ **Reward Mean 不为 0**（至少 ≥ 1e-6 量级）
- ✅ **Reward Std 有非零值**（表示存在奖励差异）
- ✅ **Reward Max/Min 不对称**（表示奖励有正有负）

---

## 后续训练建议

修改后重新训练时，应观察：

1. **奖励曲线**
   - 预期：奖励随训练逐步增加（因为策略学习）
   - 如仍为0：检查 scale 是否生效，或可再增大 10 倍至 1e-5

2. **合规率曲线**
   - 预期：合规率随时间提高（因为 RPS 目标越来越严格）
   - 符合数据：绿电占比应稳步增加

3. **市场价格**
   - 预期：价格波动反映供求变化
   - 验证：CAQ/TGC 价格应随合规率变化

---

## 代码改动汇总

| 文件 | 位置 | 修改类型 | 影响 |
|------|------|--------|------|
| `envs/vectorized_env.py` | L88-100 | compliance_reward | 合规判断改为基于绿电 |
| `envs/vectorized_env.py` | L102-109 | RewardConfig.scale | 1e-8 → 1e-6（修复奖励为0） |
| `envs/vectorized_env.py` | L660-665 | _collect_info | 合规率计算统一 |

---

## 测试文件

新增：`test_reward_fix.py`  
功能：诊断奖励值是否正常，验证修复成功
