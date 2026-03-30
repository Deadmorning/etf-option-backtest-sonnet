# -*- coding: utf-8 -*-
"""
Step 1: 使用AKShare获取创业板ETF(159915) 2026年1-3月价格数据
Step 2: 获取创业板ETF(159915)所有期权合约名称和行权价格
"""

import akshare as ak
import pandas as pd
import os
import json
from datetime import datetime

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: 获取创业板ETF(159915) 2026年1-3月日线数据")
print("=" * 60)

# ===== Step 1: ETF 日线数据 =====
etf_df = ak.fund_etf_hist_em(
    symbol="159915",
    period="daily",
    start_date="20260101",
    end_date="20260331",
    adjust=""  # 不复权
)

etf_df = etf_df.rename(columns={
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "change_pct",
    "涨跌额": "change_amt",
    "换手率": "turnover_rate"
})

etf_df["date"] = pd.to_datetime(etf_df["date"])
etf_df = etf_df.sort_values("date").reset_index(drop=True)

print(f"\n共获取到 {len(etf_df)} 条ETF日线数据")
print(f"日期范围: {etf_df['date'].min().date()} 至 {etf_df['date'].max().date()}")
print("\n前10条数据:")
print(etf_df[["date", "open", "high", "low", "close", "volume"]].head(10).to_string(index=False))
print("\n最后5条数据:")
print(etf_df[["date", "open", "high", "low", "close", "volume"]].tail(5).to_string(index=False))

# 保存ETF数据
etf_csv = os.path.join(OUTPUT_DIR, "etf_159915_daily_2026Q1.csv")
etf_df.to_csv(etf_csv, index=False, encoding="utf-8-sig")
print(f"\nETF日线数据已保存: {etf_csv}")

# ===== 也获取前复权数据供计算使用 =====
etf_qfq = ak.fund_etf_hist_em(
    symbol="159915",
    period="daily",
    start_date="20251201",  # 多取一个月供计算昨收
    end_date="20260331",
    adjust="qfq"
)
etf_qfq = etf_qfq.rename(columns={
    "日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume",
    "成交额": "turnover"
})
etf_qfq["date"] = pd.to_datetime(etf_qfq["date"])
etf_qfq = etf_qfq.sort_values("date").reset_index(drop=True)
etf_qfq_csv = os.path.join(OUTPUT_DIR, "etf_159915_qfq_ext.csv")
etf_qfq.to_csv(etf_qfq_csv, index=False, encoding="utf-8-sig")
print(f"ETF前复权数据(含12月)已保存: {etf_qfq_csv}")

print("\n" + "=" * 60)
print("Step 2: 获取创业板ETF(159915)所有期权合约名称和行权价格")
print("=" * 60)

# ===== Step 2: 获取期权合约列表 =====
option_all = ak.option_current_day_szse()

# 过滤159915
option_159915 = option_all[
    option_all["合约简称"].str.contains("159915", na=False) |
    option_all["合约代码"].str.contains("159915", na=False) |
    option_all["标的证券简称(代码)"].str.contains("159915", na=False)
].copy()

option_159915 = option_159915.reset_index(drop=True)

print(f"\n共获取到 {len(option_159915)} 个159915期权合约")
print(f"到期月份分布:")
print(option_159915["最后交易日"].value_counts().sort_index().to_string())

print("\n合约列表（名称 + 行权价 + 类型）:")
display_cols = ["合约编码", "合约代码", "合约简称", "合约类型", "行权价", "最后交易日"]
print(option_159915[display_cols].to_string(index=False))

# 保存期权合约列表
option_csv = os.path.join(OUTPUT_DIR, "option_159915_contracts.csv")
option_159915.to_csv(option_csv, index=False, encoding="utf-8-sig")
print(f"\n期权合约列表已保存: {option_csv}")

# 统计信息
call_contracts = option_159915[option_159915["合约类型"] == "认购"]
put_contracts = option_159915[option_159915["合约类型"] == "认沽"]
print(f"\n认购期权: {len(call_contracts)} 个")
print(f"认沽期权: {len(put_contracts)} 个")
print(f"行权价范围: {option_159915['行权价'].min()} - {option_159915['行权价'].max()}")

print("\n✓ Step 1 & 2 完成")
