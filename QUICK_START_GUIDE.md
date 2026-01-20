# 用户问题解答与工具更新总结

## 问题 1：合规率的计算逻辑确认

### ✅ 确认：基于购买后的绿电消费

**精确描述**：
合规率是基于 **"实际绿电消费是否满足RPS要求"** 来判断的。

```
合规判断 = 绿电消费 >= RPS要求量
         = (总消电 × 绿电占比) >= (总消电 × RPS目标比例)
```

**关键澄清**：
- 购买 CAQ/TGC 配额 **不会改变绿电消费量**（这是物理量，由企业的能源结构决定）
- 配额购买只改变财务余额（用于罚款计算）
- 所以合规率是基于 **物理绿电**，与市场交易无关

**代码位置**：
- `envs/vectorized_env.py` 第 88-100 行：`compliance_reward()` 函数
- `envs/vectorized_env.py` 第 668-672 行：`_collect_info()` 中的合规率计算

---

## 问题 2：现在的奖励函数

### 主要奖励：`profit_reward`（权重 1.0）

```python
利润 = 售电收入 - 发电成本 - CAQ交易支出 - TGC交易支出 - 违规罚款 + 盈余奖励

最终奖励 = 利润 / 平均需求量 × scale(1e-6)
```

**具体公式**：
```
收入 = 消电量 × 零售电价 (590元/MWh)
成本 = 火电消电 × 火电价(371) + 绿电消电 × 绿电价(406)
CAQ成本 = CAQ成交量 × CAQ价格
TGC成本 = TGC成交量 × TGC价格
罚款 = max(0, -余额) × 罚款单价 (600元/MWh，可配置)
奖励 = max(0, 余额) × 奖励单价 (0元/MWh，可配置)

利润 = 收入 - 成本 - CAQ成本 - TGC成本 - 罚款 + 奖励
```

**归一化**：
- 利润除以平均需求量（避免需求量变大导致奖励无限增长）
- 再乘以 scale = 1e-6（【关键修复】原值 1e-8 导致奖励为 0）

### 辅助奖励：`compliance_reward`（可选，当前权重 0）

```python
compliance = 1 if 绿电消费 >= RPS要求 else 0
```

---

## 问题 3、4 的改进

### ✅ 改进 1：移除 TensorBoard

**原因**：TensorBoard 只用于实验监控，分布式训练不需要

**删除内容**：
- `from torch.utils.tensorboard import SummaryWriter` 导入
- 所有 `writer.add_scalar()` 调用
- TensorBoard log 文件夹创建

**保留内容**：
- 奖励数据仍存储为 JSON（用于后续绘图）
- 控制台输出奖励信息

### ✅ 改进 2：新增绘图脚本 `plot_reward_from_checkpoint.py`

**功能**：
1. 输入 scenario 和 episode 编号
2. 读取对应的 checkpoint 模型
3. 运行一次完整仿真（11期：2020-2030）
4. 绘制 4 合 1 图表：
   - 奖励曲线（折线图）
   - 合规率曲线（趋势图）
   - 绿电占比曲线（变化图）
   - 统计摘要（均值、最终值等）
5. 导出 PNG 图表 + CSV 数据

**使用方法**：
```bash
# 绘制A场景第100轮的奖励
python plot_reward_from_checkpoint.py --scenario A --episode 100

# 绘制B场景第50轮的奖励
python plot_reward_from_checkpoint.py --scenario B --episode 50
```

**输出**：
- `results/A_episode_0100_reward.png` - 图表
- `results/A_episode_0100_metrics.csv` - 数据

### ✅ 改进 3：train_gpu.py 快速调试模式

**修改**：
- 默认 episodes 改为 100（从原有的更大值）
- 每 10 episodes 保存一次 checkpoint
- 训练完成后自动提示绘图命令

**快速训练命令**：
```bash
# 调试运行A场景100轮
python train_gpu.py --scenario A --num_episodes 100

# 输出示例：
# ================================================================================
# ✅ 训练完成！
#    场景: A
#    轮数: 100
#    平均奖励: 0.015
#    最终奖励: 0.022
#    耗时: 1234.5s
#
# 📊 绘制奖励曲线:
#    python plot_reward_from_checkpoint.py --scenario A --episode 100
# ================================================================================
```

---

## 快速启动指南

### 1️⃣ 训练模型（100轮调试）
```bash
python train_gpu.py --scenario A --num_episodes 100
```

预期耗时：~20-30 分钟（取决于GPU）

输出：
- `gpu_models/A_checkpoints/A_episode_0010.pt` 
- `gpu_models/A_checkpoints/A_episode_0020.pt`
- ...
- `gpu_models/A_checkpoints/A_episode_0100.pt`

### 2️⃣ 绘制奖励曲线
```bash
python plot_reward_from_checkpoint.py --scenario A --episode 100
```

预期耗时：~1-2 分钟

输出：
- `results/A_episode_0100_reward.png` ← 🎯 看这个图
- `results/A_episode_0100_metrics.csv` ← 数据表

### 3️⃣ 查看结果
图表包含：
- **奖励趋势**：是否上升（训练有效）
- **合规率**：是否提高（政策有效）
- **绿电占比**：是否增加（能源转型）
- **统计数据**：均值、最终值、范围

---

## 文件变更汇总

| 文件 | 类型 | 说明 |
|------|------|------|
| `train_gpu.py` | 修改 | 移除 TensorBoard；默认 100 episodes；完成提示 |
| `plot_reward_from_checkpoint.py` | **新增** | 绘图脚本（核心工具） |
| `QA_CLARIFICATION.md` | **新增** | 问题澄清文档 |
| `envs/vectorized_env.py` | 已修改 | 合规率/奖励修复（之前提交） |

---

## 常见问题

### Q1：为什么奖励为负？
**正常情况**。奖励 = 利润/需求量，如果成本高于收入就是负值。

### Q2：合规率一直是0怎么办？
检查RPS政策是否过于严格。初始RPS=15%，每年增1%，到2030年达30%。

### Q3：奖励一直为0是什么问题？
应该已修复（scale 从 1e-8 改为 1e-6）。如果仍为0，运行 `test_reward_fix.py` 诊断。

### Q4：模型保存在哪？
`gpu_models/{scenario}_checkpoints/{scenario}_episode_XXXX.pt`

### Q5：如何改变训练轮数？
```bash
python train_gpu.py --scenario A --num_episodes 500  # 训练500轮
python plot_reward_from_checkpoint.py --scenario A --episode 500  # 绘制第500轮
```

---

## 下一步建议

1. ✅ 运行 100 轮调试训练
2. ✅ 用绘图脚本查看奖励/合规率变化
3. ⏭️ 如果效果好，改为更多轮数（500-1000）
4. ⏭️ 对比不同场景（A/B/C）
5. ⏭️ 调整奖励函数权重（如加入 compliance_reward）

