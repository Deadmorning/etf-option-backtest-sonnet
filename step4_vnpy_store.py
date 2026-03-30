# -*- coding: utf-8 -*-
"""
Step 4: 把数据格式转成VeighNA数据库格式，存储到VeighNA数据库
- ETF日线数据 → BarData → vnpy_sqlite数据库
- 期权合约信息 → 保存为VeighNA可识别的格式
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from vnpy.trader.object import BarData, ContractData
from vnpy.trader.constant import Exchange, Interval, Product, OptionType
from vnpy.trader.database import get_database
import vnpy_sqlite

OUTPUT_DIR = "outputs"

print("=" * 60)
print("Step 4: 数据存入VeighNA数据库")
print("=" * 60)

# 初始化VeighNA SQLite数据库
# 配置数据库路径
import os
home_dir = os.path.expanduser("~")
vntrader_dir = os.path.join(home_dir, ".vntrader")
os.makedirs(vntrader_dir, exist_ok=True)

# 写入VeighNA设置（使用SQLite）
settings_file = os.path.join(vntrader_dir, "vt_setting.json")
import json

settings = {
    "database.name": "sqlite",
    "database.database": "database.db",
    "database.host": "localhost",
    "database.port": 3306,
    "database.user": "root",
    "database.password": ""
}
with open(settings_file, "w") as f:
    json.dump(settings, f, indent=2)

print(f"VeighNA设置已写入: {settings_file}")

# 获取数据库实例
db = get_database()
print(f"数据库类型: {type(db).__name__}")

# ================================================================
# 4a: ETF日线数据 → BarData → 存入VeighNA
# ================================================================
print("\n--- 4a: 存入ETF日线数据 ---")

etf_df = pd.read_csv(os.path.join(OUTPUT_DIR, "etf_159915_daily_2026Q1.csv"), parse_dates=["date"])

tz_cst = pytz.timezone("Asia/Shanghai")

etf_bars = []
for _, row in etf_df.iterrows():
    dt = row["date"].to_pydatetime()
    # VeighNA要求datetime带timezone
    dt_aware = tz_cst.localize(dt.replace(hour=15, minute=0, second=0))

    bar = BarData(
        symbol="159915",
        exchange=Exchange.SZSE,
        datetime=dt_aware,
        interval=Interval.DAILY,
        volume=float(row["volume"]),
        turnover=float(row["turnover"]) if "turnover" in row and not pd.isna(row["turnover"]) else 0.0,
        open_interest=0.0,
        open_price=float(row["open"]),
        high_price=float(row["high"]),
        low_price=float(row["low"]),
        close_price=float(row["close"]),
        gateway_name="AKShare"
    )
    etf_bars.append(bar)

# 写入数据库
db.save_bar_data(etf_bars)
print(f"✓ ETF日线数据已存入VeighNA: {len(etf_bars)} 条")

# 验证读取
overview = db.get_bar_overview()
for o in overview:
    if o.symbol == "159915":
        print(f"  验证: {o.symbol}.{o.exchange.value} | {o.interval.value} | {o.count}条 | {o.start} ~ {o.end}")

# ================================================================
# 4b: 期权合约Greeks数据 → BarData格式 → 存入VeighNA
# ================================================================
print("\n--- 4b: 存入期权合约数据（以合约代码为symbol）---")

greeks_df = pd.read_csv(os.path.join(OUTPUT_DIR, "option_159915_greeks.csv"), parse_dates=["expiry"])

# 每个合约存一条Bar（使用今日日期，close=Black76价格）
option_bars = []
today_dt = tz_cst.localize(datetime(2026, 3, 30, 15, 0, 0))

for _, row in greeks_df.iterrows():
    symbol = row["合约代码"]
    if pd.isna(row["Black76_Price"]):
        continue

    bar = BarData(
        symbol=symbol,
        exchange=Exchange.SZSE,
        datetime=today_dt,
        interval=Interval.DAILY,
        volume=0.0,
        turnover=0.0,
        open_interest=0.0,
        open_price=float(row["Black76_Price"]),
        high_price=float(row["Black76_Price"]),
        low_price=float(row["Black76_Price"]),
        close_price=float(row["Black76_Price"]),
        gateway_name="AKShare"
    )
    option_bars.append(bar)

db.save_bar_data(option_bars)
print(f"✓ 期权定价数据已存入VeighNA: {len(option_bars)} 条合约")

# ================================================================
# 4c: 同时保存扩展数据（含ETF前复权历史）
# ================================================================
print("\n--- 4c: 存入ETF前复权扩展数据（含2025年12月）---")

etf_ext = pd.read_csv(os.path.join(OUTPUT_DIR, "etf_159915_qfq_ext.csv"), parse_dates=["date"])

ext_bars = []
for _, row in etf_ext.iterrows():
    dt = row["date"].to_pydatetime()
    dt_aware = tz_cst.localize(dt.replace(hour=15, minute=0, second=0))

    bar = BarData(
        symbol="159915.QFQ",  # 前复权版本用不同symbol区分
        exchange=Exchange.SZSE,
        datetime=dt_aware,
        interval=Interval.DAILY,
        volume=float(row["volume"]),
        turnover=float(row.get("turnover", 0)) if not pd.isna(row.get("turnover", 0)) else 0.0,
        open_interest=0.0,
        open_price=float(row["open"]),
        high_price=float(row["high"]),
        low_price=float(row["low"]),
        close_price=float(row["close"]),
        gateway_name="AKShare"
    )
    ext_bars.append(bar)

db.save_bar_data(ext_bars)
print(f"✓ ETF前复权数据已存入VeighNA: {len(ext_bars)} 条")

# ================================================================
# 4d: 输出数据库概览
# ================================================================
print("\n--- VeighNA数据库概览 ---")
overview = db.get_bar_overview()
print(f"数据库共有 {len(overview)} 个数据集:")
for o in overview:
    print(f"  {o.symbol:<25} | {o.exchange.value:<6} | {o.interval.value:<7} | {o.count:>4}条 | {str(o.start)[:10]} ~ {str(o.end)[:10]}")

# 保存数据库路径供后续使用
db_info = {
    "db_path": os.path.join(vntrader_dir, "database.db"),
    "symbols": [
        {"symbol": "159915", "exchange": "SZSE", "interval": "d", "desc": "创业板ETF日线（不复权）"},
        {"symbol": "159915.QFQ", "exchange": "SZSE", "interval": "d", "desc": "创业板ETF日线（前复权）"},
    ]
}
with open(os.path.join(OUTPUT_DIR, "vnpy_db_info.json"), "w") as f:
    json.dump(db_info, f, ensure_ascii=False, indent=2)

print(f"\n✓ VeighNA数据库信息已保存: outputs/vnpy_db_info.json")
print(f"✓ 数据库文件路径: {db_info['db_path']}")
print("\n✓ Step 4 完成")
