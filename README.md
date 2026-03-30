# 创业板ETF (159915) 期权三阶段策略回测

基于 AKShare + py_vollib + VeighNA + Black-76 的完整 ETF 期权回测框架。

## 策略逻辑

### 三阶段日内策略

| 阶段 | 时间 | 条件 | 动作 |
|------|------|------|------|
| 阶段1 探索 | 09:30 | 开盘涨跌 > ±0.5% | 买入 ATM 认购/认沽 |
| 阶段2 验证 | 09:30~13:30 (每30分钟) | 价格偏离开盘 > 0.5% 反向 | 反手换仓（最多一次）|
| 阶段3 对冲 | 任意时间 | 浮盈 > 5% | 卖出 OTM 期权锁利 |
| 强平 | 14:45 | 无条件 | 平仓所有持仓 |

### 关键改进历史

- **v1**：仅开盘买入 + 收盘卖出，无日内验证
- **v2**：引入每30分钟验证，对冲阈值5%，修复账务双重计算Bug
- **v3**：动态合约换月（1月→2月→3月→4月），ATM精度0.05元档，反手截止13:30

## 文件结构

```
├── step1_fetch_etf_data.py      # AKShare 拉取 ETF 日线数据
├── step3_option_pricing.py      # Black-76 定价 + SVI 曲面校准
├── step4_vnpy_store.py          # 存入 VeighNA SQLite 数据库
├── step5_backtest_v3.py         # 主回测引擎（v3 动态换月版）
├── generate_report_v3.py        # HTML 回测报告生成
└── outputs/
    ├── backtest_report_v3.html  # 可视化回测报告
    ├── backtest_v3_perf.json    # 绩效指标 JSON
    └── svi_parameters.json      # SVI 波动率曲面参数
```

## 回测结果（v3.1）

| 指标 | 值 |
|------|-----|
| 回测周期 | 2026-01-01 ~ 2026-03-31（54交易日）|
| 初始资金 | ¥200,000 |
| 期末净值 | ¥199,806 |
| 总收益率 | -0.10% |
| 年化收益率 | -0.45% |
| 夏普比率 | -2.83 |
| 最大回撤 | -0.44% |
| 胜率 | 29.6% |
| 盈亏比 | 0.779 |

## 合约换月规则

距到期 ≤ 5 交易日时自动换月：
- 1月1日 ~ 1月20日：使用**1月**合约
- 1月21日 ~ 2月9日：使用**2月**合约
- 2月10日 ~ 3月17日：使用**3月**合约
- 3月18日 ~ ：使用**4月**合约

## 依赖环境

```bash
pip install akshare py_vollib vnpy pandas numpy scipy
```

## 数据源

- **ETF 价格**：AKShare `fund_etf_hist_em`
- **期权合约**：AKShare `option_current_em`
- **定价模型**：py_vollib Black-76
- **波动率曲面**：SVI (Stochastic Volatility Inspired) 校准
- **数据存储**：VeighNA SQLite 数据库
