# -*- coding: utf-8 -*-
"""
2026-03-30 单日回测 — 使用真实分钟数据
对比：v3.2（合成价格）vs v4优化版（真实分钟价格 + 4项优化）

数据来源: uploads/159915_20260330_intraday.csv
"""

import sys, os
import numpy as np
import pandas as pd
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/tmp/etf-option-strategy")
from py_vollib.black import black

# ── 参数配置（与v4一致）─────────────────────────────────────
RISK_FREE      = 0.018
MULTIPLIER     = 10000
COMMISSION     = 5.0
INIT_CAPITAL   = 200_000.0
OPEN_THRESH    = 0.005     # 开盘>0.5% 触发信号
REV_THRESH_V3  = 0.005     # v3 反手阈值
REV_THRESH_V4  = 0.008     # v4 反手阈值（优化3）
PROFIT_THRESH  = 0.05      # 对冲触发阈值
STOP_LOSS_PCT  = 0.35      # [OPT4] 期权止损35%
HEDGE_CONFIRM_N = 2        # [OPT5] 对冲连续确认次数
GAP_FADE_VOL_MULT = 5.0    # [OPT2] 跳空回补量能倍数阈值

TRADE_DATE = "2026-03-30"
PREV_CLOSE = 3.293         # 前一交易日(3/27)收盘价（由开盘-1.55%反推）
EXPIRY     = date(2026, 4, 22)   # 4月合约到期日（3/30已过3月换月点）

INTRADAY_TIMES = ["09:30","09:45","10:00","10:15","10:30","10:45",
                  "11:00","11:15","11:30",
                  "13:00","13:15","13:30","13:45","14:00","14:15",
                  "14:30","14:45"]
MONTH_CN = {4: "4月"}

# ── Black-76 工具函数 ─────────────────────────────────────────
def estimate_iv(flag: str, K: float, F: float, T: float) -> float:
    base = 0.52 if flag == 'c' else 0.46
    T_ref = 0.063
    term_adj = (T_ref / max(T, 0.01)) ** 0.08
    moneyness = np.log(F / K)
    skew = -0.15 * moneyness
    return max(0.15, min(base * term_adj + skew, 1.20))

def bk76(flag: str, F: float, K: float, T: float, sigma: float) -> float:
    try:
        p = black(flag, F, K, max(T, 1/365), RISK_FREE, max(sigma, 0.05))
        return max(p, 0.0001)
    except Exception:
        iv = max(0, F-K) if flag=='c' else max(0, K-F)
        return max(iv + F*sigma*np.sqrt(max(T,1/365))*0.15, 0.0001)

def T_years(expiry: date, ts_str: str) -> float:
    exp_dt = datetime(expiry.year, expiry.month, expiry.day, 15, 0)
    cur_dt = datetime.strptime(f"{TRADE_DATE} {ts_str}", "%Y-%m-%d %H:%M")
    delta  = exp_dt - cur_dt
    return max(delta.total_seconds() / (365 * 86400), 0.5/365)

def get_atm_contract(cn_type: str, spot: float, ts: str) -> dict:
    flag   = 'c' if cn_type == "认购" else 'p'
    atm_k  = round(round(spot / 0.05) * 0.05, 3)
    T      = T_years(EXPIRY, ts)
    iv     = estimate_iv(flag, atm_k, spot, T)
    premium = bk76(flag, spot, atm_k, T, iv)
    return {"type": cn_type, "flag": flag, "strike": atm_k,
            "premium": premium, "iv": iv, "T": T,
            "name": f"创业板ETF{cn_type[0]}4月{int(atm_k*1000)}"}

def get_otm_contract(cn_type: str, spot: float, ts: str) -> dict | None:
    flag = 'c' if cn_type == "认购" else 'p'
    atm_k = round(round(spot / 0.05) * 0.05, 3)
    T = T_years(EXPIRY, ts)
    if cn_type == "认购":
        otm_k = round(round((spot * 1.03) / 0.05) * 0.05, 3)
    else:
        otm_k = round(round((spot * 0.97) / 0.05) * 0.05, 3)
    if otm_k == atm_k:
        return None
    iv = estimate_iv(flag, otm_k, spot, T)
    premium = bk76(flag, spot, otm_k, T, iv)
    return {"type": cn_type, "flag": flag, "strike": otm_k,
            "premium": premium, "iv": iv, "T": T,
            "name": f"创业板ETF{cn_type[0]}4月{int(otm_k*1000)}"}

def reprice(contract: dict, open_prem: float, spot: float, ts: str) -> float:
    T  = T_years(EXPIRY, ts)
    iv = estimate_iv(contract["flag"], contract["strike"], spot, T)
    return bk76(contract["flag"], spot, contract["strike"], T, iv)

# ── 加载真实分钟数据 ──────────────────────────────────────────
df = pd.read_csv("uploads/159915_20260330_intraday.csv")
df["time"] = pd.to_datetime(df["time"])
df["ts"]   = df["time"].dt.strftime("%H:%M")

# 获取15分钟节点的真实价格
real_prices = {}
for t in INTRADAY_TIMES:
    row = df[df["ts"] == t]
    if not row.empty:
        real_prices[t] = row.iloc[0]["close"]

# 计算量能基准（10:00-11:00 均值）
avg_vol = df[(df["time"] >= "2026-03-30 10:00") &
             (df["time"] <= "2026-03-30 11:00")]["volume"].mean()
vol_0931 = df[df["ts"] == "09:31"]["volume"].values[0] if not df[df["ts"]=="09:31"].empty else 0
vol_ratio = vol_0931 / avg_vol if avg_vol > 0 else 0

print("=" * 70)
print(f"  2026-03-30 单日分析  |  使用真实分钟数据  |  前收: ¥{PREV_CLOSE}")
print("=" * 70)
print(f"\n真实15分钟价格节点:")
open_p = real_prices.get("09:30", 3.242)
print(f"{'时间':<8} {'实际价格':>8} {'偏离开盘':>10}")
for t in INTRADAY_TIMES:
    p = real_prices.get(t, None)
    if p:
        dev = (p - open_p) / open_p * 100
        flag = " ← 触发" if abs(dev) > 0.5 else ""
        print(f"  {t:<8} {p:>7.3f} {dev:>+9.2f}%{flag}")

print(f"\n量能分析:")
print(f"  09:31 成交量: {vol_0931:,.0f} 手 ({vol_ratio:.1f}x 日内均量)")
gap_fade = vol_ratio > GAP_FADE_VOL_MULT
if gap_fade:
    print(f"  ⚠ 跳空回补信号: 09:31量能 {vol_ratio:.1f}x > {GAP_FADE_VOL_MULT}x 阈值")
    print(f"  ⚠ 开盘方向可能是短暂跳空，不宜跟随")


# ── 模拟函数 ─────────────────────────────────────────────────
def run_simulation(name: str, use_entry_delay: bool, rev_thresh: float,
                   use_stop_loss: bool, hedge_confirm_n: int,
                   use_vol_filter: bool):
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"  入场: {'09:45确认' if use_entry_delay else '09:30立即'}  "
          f"反手阈值: {rev_thresh*100:.1f}%  "
          f"止损: {'35%' if use_stop_loss else '无'}  "
          f"对冲确认: {hedge_confirm_n}次  "
          f"量能过滤: {'开' if use_vol_filter else '关'}")
    print(f"{'─'*70}")

    capital = INIT_CAPITAL
    trades  = []
    change_rate = (open_p - PREV_CLOSE) / PREV_CLOSE

    # 开盘信号
    signal = None
    if change_rate > OPEN_THRESH:
        signal = "认购"
    elif change_rate < -OPEN_THRESH:
        signal = "认沽"

    print(f"\n开盘价: {open_p:.3f}  开盘涨跌: {change_rate*100:+.2f}%  "
          f"信号: {signal or '无'}")

    if not signal:
        print("→ 无信号，当日不交易")
        return capital, trades

    # 量能过滤
    if use_vol_filter and gap_fade:
        print(f"→ [量能过滤] 09:31量能{vol_ratio:.1f}x > {GAP_FADE_VOL_MULT}x，取消{signal}信号")
        return capital, trades

    long_pos  = None   # 多头持仓 {contract, open_prem, cur_price}
    short_pos = None   # 空头对冲持仓
    reversed_ = hedged = stop_loss_done = False
    hedge_confirm = 0

    for i, ts in enumerate(INTRADAY_TIMES[1:], 1):
        spot = real_prices.get(ts)
        if spot is None:
            continue
        cur_dt_str = ts

        # 重新定价
        if long_pos:
            long_pos["cur_price"] = reprice(long_pos["contract"],
                                            long_pos["open_prem"], spot, ts)
        if short_pos:
            short_pos["cur_price"] = reprice(short_pos["contract"],
                                             short_pos["open_prem"], spot, ts)

        # ── 14:45 强平
        if ts >= "14:45":
            if long_pos:
                pnl = (long_pos["cur_price"] - long_pos["open_prem"]) * MULTIPLIER - COMMISSION
                capital += pnl
                trades.append((ts, "卖出平仓", long_pos["contract"]["name"],
                               long_pos["cur_price"], pnl, "14:45强平"))
                print(f"  {ts} 【强平多】{long_pos['contract']['name']} "
                      f"P={long_pos['cur_price']:.4f} 盈亏={pnl:+.2f}")
            if short_pos:
                pnl = -(short_pos["cur_price"] - short_pos["open_prem"]) * MULTIPLIER - COMMISSION
                capital += pnl
                trades.append((ts, "买入平仓", short_pos["contract"]["name"],
                               short_pos["cur_price"], pnl, "14:45强平"))
                print(f"  {ts} 【强平空】{short_pos['contract']['name']} "
                      f"P={short_pos['cur_price']:.4f} 盈亏={pnl:+.2f}")
            break

        # ── [OPT2] 09:45 延迟入场确认
        if use_entry_delay:
            if ts == "09:45" and signal and not long_pos and not stop_loss_done:
                cur_chg = (spot - open_p) / open_p
                if (signal == "认购" and cur_chg < -OPEN_THRESH) or \
                   (signal == "认沽" and cur_chg >  OPEN_THRESH):
                    print(f"  {ts} 【放弃入场】{signal}信号已反转 "
                          f"({cur_chg*100:+.2f}% 偏离方向)")
                    signal = None
                else:
                    c = get_atm_contract(signal, spot, ts)
                    long_pos = {"contract": c, "open_prem": c["premium"],
                                "cur_price": c["premium"]}
                    # 注: 不在开仓时扣资金，只在平仓时计算净盈亏
                    trades.append((ts, "买入开仓", c["name"], c["premium"], 0,
                                   f"09:45确认{change_rate*100:+.2f}%"))
                    print(f"  {ts} 【开仓】{c['name']} K={c['strike']} "
                          f"P={c['premium']:.4f} T={c['T']*365:.0f}天 σ={c['iv']:.2f}")
        else:
            # v3: 09:30 立即入场
            if ts == "09:45" and signal and not long_pos and not stop_loss_done:
                c = get_atm_contract(signal, open_p, "09:30")
                long_pos = {"contract": c, "open_prem": c["premium"],
                            "cur_price": reprice(c, c["premium"], spot, ts)}
                # 注: 不在开仓时扣资金，只在平仓时计算净盈亏
                trades.append(("09:30", "买入开仓", c["name"], c["premium"], 0,
                               f"开盘{change_rate*100:+.2f}%"))
                print(f"  09:30 【开仓】{c['name']} K={c['strike']} "
                      f"P={c['premium']:.4f} T={c['T']*365:.0f}天 σ={c['iv']:.2f}")

        if not long_pos:
            continue

        long_pct = (long_pos["cur_price"] - long_pos["open_prem"]) / long_pos["open_prem"]

        # ── [OPT4] 止损
        loss_pct = (long_pos["open_prem"] - long_pos["cur_price"]) / long_pos["open_prem"]
        if use_stop_loss and not stop_loss_done and loss_pct > STOP_LOSS_PCT:
            pnl = (long_pos["cur_price"] - long_pos["open_prem"]) * MULTIPLIER - COMMISSION
            capital += pnl
            trades.append((ts, "止损平仓", long_pos["contract"]["name"],
                           long_pos["cur_price"], pnl,
                           f"跌幅{loss_pct*100:.1f}%>{STOP_LOSS_PCT*100:.0f}%"))
            print(f"  {ts} 【止损】{long_pos['contract']['name']} "
                  f"跌幅{loss_pct*100:.1f}% 亏损={pnl:.2f}")
            if short_pos:
                pnl_s = -(short_pos["cur_price"]-short_pos["open_prem"])*MULTIPLIER - COMMISSION
                capital += pnl_s
                trades.append((ts, "买入平仓", short_pos["contract"]["name"],
                               short_pos["cur_price"], pnl_s, "止损同步平对冲"))
                short_pos = None
            long_pos = None
            stop_loss_done = True
            hedge_confirm  = 0
            continue

        # ── [OPT3] 反手验证
        if not reversed_ and ts <= "13:30":
            chg = (spot - open_p) / open_p
            rev_needed = (long_pos["contract"]["flag"] == 'c' and chg < -rev_thresh) or \
                         (long_pos["contract"]["flag"] == 'p' and chg >  rev_thresh)
            if rev_needed:
                pnl = (long_pos["cur_price"]-long_pos["open_prem"])*MULTIPLIER - COMMISSION
                capital += pnl
                trades.append((ts, "反手平仓", long_pos["contract"]["name"],
                               long_pos["cur_price"], pnl, f"偏离{chg*100:.2f}%反手"))
                print(f"  {ts} 【反手平仓】{long_pos['contract']['name']} "
                      f"盈亏={pnl:+.2f}  (偏离{chg*100:+.2f}%)")
                new_cn = "认沽" if long_pos["contract"]["flag"] == 'c' else "认购"
                nc = get_atm_contract(new_cn, spot, ts)
                long_pos = {"contract": nc, "open_prem": nc["premium"],
                            "cur_price": nc["premium"]}
                # 注: 反手开仓不扣资金，平仓时结算净盈亏
                trades.append((ts, "反手开仓", nc["name"], nc["premium"], 0,
                               f"反手→{new_cn}"))
                print(f"  {ts} 【反手开仓】{nc['name']} K={nc['strike']} "
                      f"P={nc['premium']:.4f}")
                reversed_ = True
                hedge_confirm = 0

        # ── [OPT5] 对冲连续确认
        if not hedged and long_pos:
            if long_pct > PROFIT_THRESH:
                hedge_confirm += 1
                if hedge_confirm >= hedge_confirm_n:
                    hc = get_otm_contract(long_pos["contract"]["type"], spot, ts)
                    if hc:
                        short_pos = {"contract": hc, "open_prem": hc["premium"],
                                     "cur_price": hc["premium"]}
                        income = hc["premium"] * MULTIPLIER - COMMISSION
                        # 注: 对冲开仓收入在平仓时一并结算，不提前计入资金
                        trades.append((ts, "卖出对冲", hc["name"], hc["premium"], 0,
                                       f"连续{hedge_confirm}次浮盈{long_pct*100:.1f}%→对冲"))
                        print(f"  {ts} 【对冲】卖{hc['name']} K={hc['strike']} "
                              f"P={hc['premium']:.4f} 收入={income:.2f}")
                        hedged = True
            else:
                hedge_confirm = 0

    day_pnl = capital - INIT_CAPITAL
    print(f"\n  当日结果: 盈亏 {day_pnl:+.2f}  净值 {capital:,.2f}")
    print(f"  交易笔数: {len(trades)}")
    return capital, trades


# ── 对比运行两种策略 ──────────────────────────────────────────
print(f"\n{'='*70}")
print("  策略对比：v3.2（合成价格基线）vs v4优化（真实分钟数据）")
print(f"{'='*70}")

_, trades_v3 = run_simulation(
    name="【基准】v3.2 策略 — 真实分钟数据 + 原始参数",
    use_entry_delay=False,
    rev_thresh=REV_THRESH_V3,
    use_stop_loss=False,
    hedge_confirm_n=1,
    use_vol_filter=False,
)

_, trades_v4 = run_simulation(
    name="【优化】v4 策略 — 真实分钟数据 + 4项优化",
    use_entry_delay=True,
    rev_thresh=REV_THRESH_V4,
    use_stop_loss=True,
    hedge_confirm_n=HEDGE_CONFIRM_N,
    use_vol_filter=True,
)

# ── 分钟级价格走势总结 ──────────────────────────────────────
print(f"\n{'='*70}")
print("  3/30 日内价格走势关键节点（真实数据）")
print(f"{'='*70}")
phases = [
    ("09:30", "开盘"),
    ("09:31", "1分钟后"),
    ("09:34", None),
    ("09:45", "15分确认"),
    ("10:15", "低点区间"),
    ("10:24", "日内低点"),
    ("11:30", "上午收"),
    ("13:00", "下午开"),
    ("13:30", "反手截止"),
    ("14:45", "强平"),
    ("15:00", "收盘"),
]
print(f"\n  {'时间':<10} {'价格':>8} {'偏离开盘':>10} {'备注'}")
for ts, note in phases:
    row = df[df["ts"] == ts]
    if not row.empty:
        p = row.iloc[0]["close"]
        dev = (p - open_p) / open_p * 100
        note_str = f"  {note}" if note else ""
        print(f"  {ts:<10} {p:>8.3f} {dev:>+9.2f}%{note_str}")

print(f"\n  日内振幅: {df['high'].max():.3f} ~ {df['low'].min():.3f} "
      f"= {(df['high'].max()-df['low'].min())/open_p*100:.2f}%")
print(f"  收盘涨跌: {(real_prices.get('15:00',3.264)-PREV_CLOSE)/PREV_CLOSE*100:+.2f}%\n")
