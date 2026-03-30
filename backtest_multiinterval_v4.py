# -*- coding: utf-8 -*-
"""
多时间间隔对比回测 v4 (3/19 ~ 3/30 真实1分钟数据)
三个版本：5分钟 / 10分钟 / 15分钟 检查间隔
"""

import sys, os, json, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")
import glob

sys.path.insert(0, "/tmp/etf-option-strategy")
from py_vollib.black import black

OUTPUT_DIR = "outputs"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 三种时间间隔配置 ───────────────────────────────────────────
INTERVAL_CONFIGS = {
    "5min": {
        "label": "5分钟",
        "entry_time": "09:45",
        "clear_time": "14:45",
        "times": [
            "09:30","09:35","09:40","09:45","09:50","09:55",
            "10:00","10:05","10:10","10:15","10:20","10:25",
            "10:30","10:35","10:40","10:45","10:50","10:55",
            "11:00","11:05","11:10","11:15","11:20","11:25","11:30",
            "13:00","13:05","13:10","13:15","13:20","13:25",
            "13:30","13:35","13:40","13:45","13:50","13:55",
            "14:00","14:05","14:10","14:15","14:20","14:25",
            "14:30","14:35","14:40","14:45"
        ]
    },
    "10min": {
        "label": "10分钟",
        "entry_time": "09:50",   # 09:45不在10min格点，用09:50
        "clear_time": "14:40",
        "times": [
            "09:30","09:40","09:50","10:00","10:10","10:20",
            "10:30","10:40","10:50","11:00","11:10","11:20","11:30",
            "13:00","13:10","13:20","13:30","13:40","13:50",
            "14:00","14:10","14:20","14:30","14:40"
        ]
    },
    "15min": {
        "label": "15分钟",
        "entry_time": "09:45",
        "clear_time": "14:45",
        "times": [
            "09:30","09:45","10:00","10:15","10:30","10:45",
            "11:00","11:15","11:30",
            "13:00","13:15","13:30","13:45","14:00","14:15",
            "14:30","14:45"
        ]
    },
}

# ── v4策略核心参数 ─────────────────────────────────────────────
RISK_FREE      = 0.018
MULTIPLIER     = 10000
COMMISSION     = 5.0
INIT_CAPITAL   = 200_000.0
OPEN_THRESH    = 0.005
REV_THRESH     = 0.008
PROFIT_THRESH  = 0.05
ROLL_DAYS      = 5
STOP_LOSS_PCT  = 0.35
HEDGE_CONFIRM_N= 2
REV_CUTOFF_TIME= "13:30"

MONTH_CN = {1:"1月",2:"2月",3:"3月",4:"4月",5:"5月",
            6:"6月",7:"7月",8:"8月",9:"9月",10:"10月",
            11:"11月",12:"12月"}

EXPIRY_SCHEDULE = {
    (2026, 3): date(2026, 3, 25),
    (2026, 4): date(2026, 4, 22),
    (2026, 5): date(2026, 5, 27),
    (2026, 6): date(2026, 6, 24),
    (2026, 9): date(2026, 9, 23),
}

def get_expiry(year, month):
    return EXPIRY_SCHEDULE.get((year, month), date(year, month, 28))

def trading_days_to_expiry(trade_date, expiry, trading_set):
    d, cnt = trade_date, 0
    while d < expiry:
        d += timedelta(days=1)
        if d in trading_set: cnt += 1
    return cnt

def choose_expiry(trade_date, trading_set):
    curr_exp = get_expiry(trade_date.year, trade_date.month)
    td = trading_days_to_expiry(trade_date, curr_exp, trading_set)
    if td <= ROLL_DAYS:
        nm = trade_date.month + 1 if trade_date.month < 12 else 1
        ny = trade_date.year if trade_date.month < 12 else trade_date.year + 1
        return get_expiry(ny, nm)
    return curr_exp

def estimate_iv(flag, K, F, T):
    base = 0.52 if flag == 'c' else 0.46
    term_adj = (0.063 / max(T, 0.01)) ** 0.08
    skew = -0.15 * np.log(F / K)
    return max(0.15, min(base * term_adj + skew, 1.20))

def bk76(flag, F, K, T, r=RISK_FREE, sigma=0.35):
    try:
        p = black(flag, F, K, max(T, 1/365), r, max(sigma, 0.05))
        return max(p, 0.0001)
    except Exception:
        iv = max(0, F-K) if flag=='c' else max(0, K-F)
        return max(iv + F*sigma*np.sqrt(max(T,1/365))*0.15, 0.0001)

def T_years(expiry, current_dt):
    exp_dt = datetime(expiry.year, expiry.month, expiry.day, 15, 0)
    return max((exp_dt - current_dt).total_seconds() / (365*86400), 0.5/365)

class DCB:
    """DynamicContractBuilder（简化版）"""
    @staticmethod
    def build(cn_type, spot, expiry):
        flag = 'c' if cn_type == "认购" else 'p'
        month = MONTH_CN[expiry.month]
        atm_k = round(round(spot / 0.05) * 0.05, 3)
        res = []
        for i in range(-4, 5):
            K = round(atm_k + i*0.05, 3)
            if K <= 0: continue
            code = f"159915{cn_type[0]}{expiry.strftime('%y%m')}{int(K*1000):05d}"
            res.append({"code": code,
                        "name": f"创业板ETF{cn_type[0]}{month}{int(K*1000)}",
                        "strike": K, "expiry": expiry.strftime("%Y-%m-%d"),
                        "type": cn_type, "flag": flag})
        return res

    @staticmethod
    def get_atm(cn_type, spot, expiry, cur_dt):
        contracts = DCB.build(cn_type, spot, expiry)
        best = min(contracts, key=lambda c: abs(c["strike"]-spot))
        T = T_years(expiry, cur_dt)
        iv = estimate_iv(best["flag"], best["strike"], spot, T)
        best["iv"] = iv
        best["premium"] = bk76(best["flag"], spot, best["strike"], T, RISK_FREE, iv)
        return best

    @staticmethod
    def get_otm(cn_type, spot, expiry, cur_dt, otm_pct=0.03):
        contracts = DCB.build(cn_type, spot, expiry)
        if cn_type == "认购":
            target = spot * (1 + otm_pct)
            cands  = [c for c in contracts if c["strike"] > spot]
        else:
            target = spot * (1 - otm_pct)
            cands  = [c for c in contracts if c["strike"] < spot]
        if not cands: cands = contracts
        best = min(cands, key=lambda c: abs(c["strike"]-target))
        atm_k = DCB.get_atm(cn_type, spot, expiry, cur_dt)["strike"]
        if best["strike"] == atm_k: return None
        T = T_years(expiry, cur_dt)
        iv = estimate_iv(best["flag"], best["strike"], spot, T)
        best["iv"] = iv
        best["premium"] = bk76(best["flag"], spot, best["strike"], T, RISK_FREE, iv)
        return best

class Position:
    def __init__(self, contract, open_premium, open_time, direction="long"):
        self.contract     = contract
        self.open_premium = open_premium
        self.current_price= open_premium
        self.open_time    = open_time
        self.direction    = direction

    def pnl(self):
        sign = 1 if self.direction == "long" else -1
        return sign * (self.current_price - self.open_premium) * MULTIPLIER

    def pnl_pct(self):
        return (self.current_price - self.open_premium) / self.open_premium

    def reprice(self, spot, cur_dt):
        exp = datetime.strptime(self.contract["expiry"], "%Y-%m-%d").date()
        T   = T_years(exp, cur_dt)
        iv  = estimate_iv(self.contract["flag"], self.contract["strike"], spot, T)
        self.current_price = bk76(self.contract["flag"], spot,
                                  self.contract["strike"], T, RISK_FREE, iv)


def get_intraday_prices(date_str, min_df, open_p, intraday_times):
    """从真实1分钟数据提取指定时间点的价格"""
    time_index = min_df.index.tolist()
    result = []
    for ts in intraday_times:
        if ts == "09:30":
            price = open_p
        else:
            candidates = [t for t in time_index if t <= ts]
            price = float(min_df.loc[max(candidates), "close"]) if candidates else open_p
        result.append((ts, price))
    return result


def run_backtest(interval_key: str, etf_df: pd.DataFrame,
                 minute_data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """运行单个间隔配置的回测，返回(daily_df, trades_df, final_capital)"""
    cfg        = INTERVAL_CONFIGS[interval_key]
    times      = cfg["times"]
    entry_time = cfg["entry_time"]
    clear_time = cfg["clear_time"]
    label      = cfg["label"]

    trading_set = set(etf_df["date_obj"])
    capital     = INIT_CAPITAL
    daily_results = []
    all_trades    = []

    def _trade(action, pos, ts, date_str, reason, pnl=0.0, expiry_info=""):
        all_trades.append({
            "date": date_str, "time": ts, "action": action,
            "option_name": pos.contract["name"],
            "option_type": pos.contract["type"],
            "direction":   pos.direction,
            "strike":      pos.contract["strike"],
            "expiry":      pos.contract["expiry"],
            "price":       round(pos.current_price, 6),
            "pnl":         round(pnl, 2),
            "reason":      reason,
            "expiry_info": expiry_info,
        })

    for idx in range(1, len(etf_df)):
        bar       = etf_df.iloc[idx]
        prev_bar  = etf_df.iloc[idx-1]
        date_str  = bar["date"]
        trade_dt  = bar["date_obj"]
        open_p    = bar["open"]
        prev_close= prev_bar["close"]

        expiry    = choose_expiry(trade_dt, trading_set)
        td_left   = trading_days_to_expiry(trade_dt, expiry, trading_set)
        exp_label = f"{MONTH_CN[expiry.month]}合约(剩{td_left}日)"

        min_df   = minute_data.get(date_str)
        intraday = get_intraday_prices(date_str, min_df, open_p, times) if min_df is not None else []

        # 若无真实数据则跳过（本次全部有真实数据）
        if not intraday:
            daily_results.append({
                "date": date_str, "etf_open": open_p, "etf_close": bar["close"],
                "etf_chg": round((bar["close"]-prev_close)/prev_close*100,3),
                "signal": "无", "daily_pnl": 0.0, "daily_return": 0.0,
                "total_value": round(capital, 2),
                "reversed": False, "hedged": False, "stop_loss": False,
                "expiry_label": exp_label,
            })
            continue

        positions: List[Position] = []
        hedged = reversed_ = stop_loss_triggered = False
        hedge_confirm = 0
        capital_at_open = capital

        change_rate = (open_p - prev_close) / prev_close
        signal = None
        if change_rate > OPEN_THRESH:   signal = "认购"
        elif change_rate < -OPEN_THRESH: signal = "认沽"

        for ts, spot in intraday[1:]:
            cur_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")
            for p in positions: p.reprice(spot, cur_dt)

            # 强平
            if ts >= clear_time:
                for p in positions[:]:
                    pnl = p.pnl() - COMMISSION
                    act = "卖出平仓" if p.direction=="long" else "买入平仓"
                    _trade(act, p, ts, date_str, f"{clear_time}强平", pnl, exp_label)
                positions = []
                break

            # 入场确认
            if ts == entry_time and signal and not positions and not stop_loss_triggered:
                cur_chg = (spot - open_p) / open_p
                abandoned = (signal=="认购" and cur_chg < -OPEN_THRESH) or \
                            (signal=="认沽" and cur_chg >  OPEN_THRESH)
                if abandoned:
                    signal = None
                else:
                    open_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")
                    c = DCB.get_atm(signal, spot, expiry, open_dt)
                    pos = Position(c, c["premium"], ts, "long")
                    positions.append(pos)
                    _trade("买入开仓", pos, ts, date_str,
                           f"{entry_time}确认{'↑' if signal=='认购' else '↓'}{change_rate*100:.2f}%",
                           expiry_info=exp_label)

            if not positions: continue

            # 止损
            lp = next((p for p in positions if p.direction=="long"), None)
            if lp and not stop_loss_triggered:
                loss_pct = (lp.open_premium - lp.current_price) / lp.open_premium
                if loss_pct > STOP_LOSS_PCT:
                    pnl_sl = lp.pnl() - COMMISSION
                    _trade("止损平仓", lp, ts, date_str,
                           f"止损:{loss_pct*100:.1f}%", pnl_sl, exp_label)
                    positions.remove(lp)
                    sp = next((p for p in positions if p.direction=="short"), None)
                    if sp:
                        _trade("买入平仓", sp, ts, date_str, "止损平对冲",
                               sp.pnl()-COMMISSION, exp_label)
                        positions.remove(sp)
                    stop_loss_triggered = True
                    hedge_confirm = 0
                    continue

            # 反手（截止 REV_CUTOFF_TIME）
            lp = next((p for p in positions if p.direction=="long"), None)
            if lp and not reversed_ and ts <= REV_CUTOFF_TIME:
                chg = (spot - open_p) / open_p
                rev_needed = (lp.contract["flag"]=='c' and chg < -REV_THRESH) or \
                             (lp.contract["flag"]=='p' and chg >  REV_THRESH)
                if rev_needed:
                    pnl_c = lp.pnl() - COMMISSION
                    _trade("反手平仓", lp, ts, date_str,
                           f"偏离{chg*100:.2f}%反手", pnl_c, exp_label)
                    positions.remove(lp)
                    new_cn = "认沽" if lp.contract["flag"]=='c' else "认购"
                    nc = DCB.get_atm(new_cn, spot, expiry, cur_dt)
                    np_ = Position(nc, nc["premium"], ts, "long")
                    positions.append(np_)
                    _trade("反手开仓", np_, ts, date_str, f"反手→{new_cn}", expiry_info=exp_label)
                    reversed_ = True
                    hedge_confirm = 0

            # 对冲（连续N次确认）
            if not hedged:
                lp = next((p for p in positions if p.direction=="long"), None)
                if lp and lp.pnl_pct() > PROFIT_THRESH:
                    hedge_confirm += 1
                    if hedge_confirm >= HEDGE_CONFIRM_N:
                        hc = DCB.get_otm(lp.contract["type"], spot, expiry, cur_dt, 0.03)
                        if hc:
                            hp = Position(hc, hc["premium"], ts, "short")
                            positions.append(hp)
                            _trade("卖出对冲", hp, ts, date_str,
                                   f"连续{hedge_confirm}次浮盈→对冲", pnl=0, expiry_info=exp_label)
                            hedged = True
                else:
                    hedge_confirm = 0

        day_pnl = sum(t["pnl"] for t in all_trades
                      if t["date"]==date_str and t["pnl"]!=0)
        capital += day_pnl
        ret = day_pnl / capital_at_open if capital_at_open else 0.0
        daily_results.append({
            "date":          date_str,
            "etf_open":      open_p,
            "etf_close":     bar["close"],
            "etf_chg":       round((bar["close"]-prev_close)/prev_close*100, 3),
            "signal":        signal or "无",
            "reversed":      reversed_,
            "hedged":        hedged,
            "stop_loss":     stop_loss_triggered,
            "daily_pnl":     round(day_pnl, 2),
            "daily_return":  round(ret, 6),
            "total_value":   round(capital, 2),
            "expiry_label":  exp_label,
        })

    return pd.DataFrame(daily_results), pd.DataFrame(all_trades), capital


# ── 数据加载 ──────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "159915_*_1min.csv")))
date_strs = []
for f in files:
    base = os.path.basename(f)
    ds = base.split("_")[1]
    date_strs.append(f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}")

print(f"找到 {len(date_strs)} 个交易日: {date_strs[0]} ~ {date_strs[-1]}")

# 构建日线
def build_daily(date_strs):
    rows = []
    for ds in date_strs:
        fname = os.path.join(UPLOAD_DIR, f"159915_{ds.replace('-','')}_1min.csv")
        if not os.path.exists(fname): continue
        df = pd.read_csv(fname, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        rows.append({
            "date": ds, "date_obj": date.fromisoformat(ds),
            "open":  df["open"].iloc[0],
            "high":  df["high"].max(),
            "low":   df["low"].min(),
            "close": df["close"].iloc[-1],
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

etf_df = build_daily(date_strs)

# 加载分钟数据
minute_data = {}
for ds in date_strs:
    fname = os.path.join(UPLOAD_DIR, f"159915_{ds.replace('-','')}_1min.csv")
    df = pd.read_csv(fname, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["time_str"] = pd.to_datetime(df["day"]).dt.strftime("%H:%M")
    minute_data[ds] = df.set_index("time_str")

print(f"分钟数据已加载，日线共 {len(etf_df)} 条\n")

# ── 三版本回测 ────────────────────────────────────────────────
results = {}
for key in ["5min", "10min", "15min"]:
    cfg = INTERVAL_CONFIGS[key]
    print(f"{'='*50}")
    print(f"运行 [{cfg['label']}间隔] 版本  ({len(cfg['times'])} 个检查点, 入场={cfg['entry_time']})")
    daily_df, trades_df, final_cap = run_backtest(key, etf_df, minute_data)
    n_open  = (trades_df["action"]=="买入开仓").sum() if len(trades_df) else 0
    n_rev   = (trades_df["action"]=="反手开仓").sum() if len(trades_df) else 0
    n_hedge = (trades_df["action"]=="卖出对冲").sum() if len(trades_df) else 0
    n_sl    = (trades_df["action"]=="止损平仓").sum() if len(trades_df) else 0
    total_ret = (final_cap - INIT_CAPITAL) / INIT_CAPITAL
    n_days = len(daily_df)
    rets = daily_df["daily_return"]
    sharpe = (rets.mean()-RISK_FREE/250)/rets.std()*np.sqrt(250) if rets.std()>0 else 0
    cum = (1+rets).cumprod()
    max_dd = ((cum-cum.cummax())/cum.cummax()).min()
    win_rate = (daily_df["daily_pnl"]>0).mean()
    results[key] = {
        "label":      cfg["label"],
        "n_checkpoints": len(cfg["times"]),
        "entry_time": cfg["entry_time"],
        "daily_df":   daily_df,
        "trades_df":  trades_df,
        "final_cap":  final_cap,
        "total_ret":  total_ret,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "win_rate":   win_rate,
        "n_open":     int(n_open),
        "n_rev":      int(n_rev),
        "n_hedge":    int(n_hedge),
        "n_sl":       int(n_sl),
        "n_days":     n_days,
    }
    print(f"  总收益: {total_ret*100:+.2f}%  夏普: {sharpe:+.2f}  最大回撤: {max_dd*100:.2f}%")
    print(f"  胜率: {win_rate*100:.0f}%  开仓:{n_open} 反手:{n_rev} 对冲:{n_hedge} 止损:{n_sl}")

# 保存JSON汇总
summary = {k: {kk: vv for kk,vv in v.items()
               if kk not in ("daily_df","trades_df")} for k,v in results.items()}
with open(os.path.join(OUTPUT_DIR, "backtest_multiinterval_perf.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ── 生成对比HTML报告 ──────────────────────────────────────────
print("\n生成对比报告...")

keys = ["5min", "10min", "15min"]
colors = {"5min": "#f59e0b", "10min": "#3b82f6", "15min": "#22c55e"}
dates_list = results["15min"]["daily_df"]["date"].tolist()

# 净值序列（各版本）
nav_series = {}
for k in keys:
    df = results[k]["daily_df"]
    nav_series[k] = [(v/INIT_CAPITAL) for v in df["total_value"].tolist()]

# 每日盈亏
pnl_series = {k: results[k]["daily_df"]["daily_pnl"].tolist() for k in keys}

# 指标卡数据
def mc(v, fmt="+.2f", suffix=""):
    cls = "pos" if v > 0 else "neg" if v < 0 else "neu"
    return f"<span class='{cls}'>{v:{fmt}}{suffix}</span>"

# 每日对比表格行
daily_rows = ""
for i, date_str in enumerate(dates_list):
    cells = f"<td>{date_str}</td>"
    for k in keys:
        df = results[k]["daily_df"]
        row = df[df["date"]==date_str]
        if row.empty:
            cells += "<td colspan='2'>-</td>"
            continue
        r = row.iloc[0]
        pnl_c = "#22c55e" if r["daily_pnl"]>0 else "#ef4444" if r["daily_pnl"]<0 else "#94a3b8"
        flags = ""
        if r.get("reversed"):  flags += "<span class='badge sell'>反</span>"
        if r.get("hedged"):    flags += "<span class='badge'>冲</span>"
        if r.get("stop_loss"): flags += "<span class='badge sell'>损</span>"
        cells += (f"<td style='color:{pnl_c}'>{r['daily_pnl']:+.2f}</td>"
                  f"<td>{flags if flags else '-'}</td>")
    daily_rows += f"<tr>{cells}</tr>\n"

# 图表数据
nav_js = {k: json.dumps([round(v,6) for v in nav_series[k]]) for k in keys}
pnl_js = {k: json.dumps([round(v,2) for v in pnl_series[k]]) for k in keys}

# 各版本交易记录表
def make_trades_table(k):
    trades_df = results[k]["trades_df"]
    if len(trades_df) == 0:
        return "<tr><td colspan='7' style='text-align:center;color:#64748b'>无交易记录</td></tr>"
    rows = ""
    for _, t in trades_df.iterrows():
        pnl_str = f"<span style='color:{'#22c55e' if t['pnl']>0 else '#ef4444' if t['pnl']<0 else '#94a3b8'}'>{t['pnl']:+.2f}</span>" if t['pnl']!=0 else "-"
        rows += (f"<tr><td>{t['date']}</td><td>{t['time']}</td>"
                 f"<td><span class='badge {'buy' if '买' in t['action'] else 'sell'}'>"
                 f"{t['action']}</span></td>"
                 f"<td>{t['option_name']}</td><td>{t['strike']}</td>"
                 f"<td>{t['price']:.4f}</td><td>{pnl_str}</td>"
                 f"<td style='color:#94a3b8;font-size:11px'>{t.get('reason','')}</td></tr>")
    return rows

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>v4策略多时间间隔对比回测 (3/20~3/30)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0f172a;color:#e2e8f0;font-size:13px;line-height:1.6}}
.container{{max-width:1440px;margin:0 auto;padding:24px}}
h1{{font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:4px}}
.subtitle{{color:#64748b;font-size:12px;margin-bottom:20px}}
.grid{{display:grid;gap:14px}}
.g3{{grid-template-columns:repeat(3,1fr)}}
.g1{{grid-template-columns:1fr}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px}}
.card h2{{font-size:13px;color:#94a3b8;margin-bottom:12px;font-weight:500;
          display:flex;align-items:center;gap:8px}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
.metric{{text-align:center}}
.metric .val{{font-size:24px;font-weight:700}}
.metric .sub{{font-size:11px;color:#64748b;margin-top:3px}}
.pos{{color:#22c55e}}.neg{{color:#ef4444}}.neu{{color:#94a3b8}}
.chart-wrap{{position:relative;height:200px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#64748b;font-weight:500;padding:8px 6px;border-bottom:1px solid #334155;text-align:left}}
td{{padding:6px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#334155}}
.badge{{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;
        background:#3b4a6b;color:#93c5fd}}
.badge.buy{{background:#14532d;color:#4ade80}}
.badge.sell{{background:#450a0a;color:#f87171}}
.tabs{{display:flex;gap:4px;margin-bottom:12px}}
.tab{{padding:5px 14px;border-radius:6px;cursor:pointer;font-size:12px;
      background:#334155;color:#94a3b8;border:none;transition:all .2s}}
.tab.active{{background:#3b82f6;color:#fff}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
.compare-header{{background:#1e293b;padding:8px 12px;border-radius:6px;
                 margin-bottom:12px;font-size:11px;color:#64748b}}
</style>
</head>
<body>
<div class="container">
  <h1>创业板ETF期权 v4策略 — 多时间间隔对比回测</h1>
  <div class="subtitle">
    回测周期: {date_strs[1]} ~ {date_strs[-1]} (7个交易日, 真实1分钟数据) &nbsp;|&nbsp;
    三版本: 5分钟 / 10分钟 / 15分钟 检查间隔
  </div>

  <!-- 三版本核心指标对比 -->
  <div class="grid g3" style="margin-bottom:14px">
"""

for k in keys:
    r = results[k]
    col = colors[k]
    tr_class = "pos" if r["total_ret"]>=0 else "neg"
    html += f"""
    <div class="card">
      <h2><span class="dot" style="background:{col}"></span>{r['label']}间隔
          <small style="color:#64748b;font-size:11px">({r['n_checkpoints']}个检查点 / 入场={r['entry_time']})</small>
      </h2>
      <div class="grid" style="grid-template-columns:repeat(2,1fr);gap:8px">
        <div class="metric">
          <div class="val {tr_class}">{r['total_ret']*100:+.2f}%</div>
          <div class="sub">总收益</div>
        </div>
        <div class="metric">
          <div class="val {'pos' if r['sharpe']>=0 else 'neg'}">{r['sharpe']:+.2f}</div>
          <div class="sub">夏普比率</div>
        </div>
        <div class="metric">
          <div class="val neg">{r['max_dd']*100:.2f}%</div>
          <div class="sub">最大回撤</div>
        </div>
        <div class="metric">
          <div class="val">{r['win_rate']*100:.0f}%</div>
          <div class="sub">胜率</div>
        </div>
      </div>
      <div style="margin-top:10px;padding-top:10px;border-top:1px solid #334155;
                  display:flex;justify-content:space-around;font-size:11px;color:#64748b">
        <span>开仓 <b style="color:#e2e8f0">{r['n_open']}</b></span>
        <span>反手 <b style="color:#e2e8f0">{r['n_rev']}</b></span>
        <span>对冲 <b style="color:#e2e8f0">{r['n_hedge']}</b></span>
        <span>止损 <b style="color:#e2e8f0">{r['n_sl']}</b></span>
      </div>
    </div>"""

html += f"""
  </div>

  <!-- 净值曲线对比 -->
  <div class="grid g1" style="margin-bottom:14px">
    <div class="card">
      <h2>三版本净值曲线对比</h2>
      <div class="chart-wrap" style="height:240px"><canvas id="navChart"></canvas></div>
    </div>
  </div>

  <!-- 每日盈亏对比 -->
  <div class="grid g1" style="margin-bottom:14px">
    <div class="card">
      <h2>每日盈亏对比 (元)</h2>
      <div class="chart-wrap" style="height:220px"><canvas id="pnlChart"></canvas></div>
    </div>
  </div>

  <!-- 每日明细对比表 -->
  <div class="card" style="margin-bottom:14px">
    <h2>每日盈亏明细对比</h2>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th rowspan="2">日期</th>
          {"".join(f'<th colspan="2" style="border-left:1px solid #334155"><span class="dot" style="background:{colors[k]}"></span> {results[k]["label"]}间隔</th>' for k in keys)}
        </tr>
        <tr>
          {"".join('<th style="border-left:1px solid #334155">日盈亏</th><th>操作</th>' for _ in keys)}
        </tr>
      </thead>
      <tbody>{daily_rows}</tbody>
    </table>
    </div>
  </div>

  <!-- 各版本交易记录 -->
  <div class="card">
    <h2>交易记录详情</h2>
    <div class="tabs">
      <button class="tab active" onclick="showTab('t5')">5分钟间隔</button>
      <button class="tab" onclick="showTab('t10')">10分钟间隔</button>
      <button class="tab" onclick="showTab('t15')">15分钟间隔</button>
    </div>"""

for k, tid in [("5min","t5"),("10min","t10"),("15min","t15")]:
    active = "active" if k=="5min" else ""
    html += f"""
    <div id="{tid}" class="tab-content {active}">
      <div style="overflow-x:auto;max-height:360px;overflow-y:auto">
      <table>
        <thead><tr>
          <th>日期</th><th>时间</th><th>操作</th><th>合约名称</th>
          <th>行权价</th><th>权利金</th><th>盈亏</th><th>原因</th>
        </tr></thead>
        <tbody>{make_trades_table(k)}</tbody>
      </table>
      </div>
    </div>"""

html += f"""
  </div>
</div>

<script>
const dates = {json.dumps(dates_list)};
const navs = {{
  "5min":  {nav_js["5min"]},
  "10min": {nav_js["10min"]},
  "15min": {nav_js["15min"]},
}};
const pnls = {{
  "5min":  {pnl_js["5min"]},
  "10min": {pnl_js["10min"]},
  "15min": {pnl_js["15min"]},
}};
const colors = {{"5min":"{colors['5min']}","10min":"{colors['10min']}","15min":"{colors['15min']}"}};
const labels = {{"5min":"5分钟","10min":"10分钟","15min":"15分钟"}};

const axOpts = {{
  x:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:10}}}}}},
  y:{{grid:{{color:'#334155'}},ticks:{{color:'#64748b',font:{{size:10}}}}}}
}};

new Chart(document.getElementById('navChart'),{{
  type:'line',
  data:{{labels:dates, datasets:Object.keys(navs).map(k=>
    ({{label:labels[k], data:navs[k],
      borderColor:colors[k], backgroundColor:colors[k]+'18',
      fill:true, tension:0.3, pointRadius:3, borderWidth:2,
      pointBackgroundColor:colors[k], pointHoverRadius:5
    }})
  )}},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:true,labels:{{color:'#94a3b8',font:{{size:11}}}}}}}},
    scales:{{...axOpts,
      y:{{...axOpts.y,
         ticks:{{...axOpts.y.ticks,callback:v=>(v*100).toFixed(2)+'%'}}}}}}
  }}
}});

const pnlDatasets = Object.keys(pnls).map((k,i)=>
  ({{label:labels[k], data:pnls[k],
    backgroundColor:colors[k]+'99', borderColor:colors[k],
    borderWidth:1
  }})
);
new Chart(document.getElementById('pnlChart'),{{
  type:'bar',
  data:{{labels:dates, datasets:pnlDatasets}},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:true,labels:{{color:'#94a3b8',font:{{size:11}}}}}}}},
    scales:{{...axOpts}}
  }}
}});

function showTab(id){{
  document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(e=>e.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

report_path = os.path.join(OUTPUT_DIR, "backtest_multiinterval_v4.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nHTML报告已生成: {report_path}")
print("\n" + "═"*52)
print("       三版本汇总对比")
print("═"*52)
print(f"{'指标':<18} {'5分钟':>10} {'10分钟':>10} {'15分钟':>10}")
print("-"*52)
for metric, fmt, suffix in [
    ("总收益率", "+.2f", "%"),
    ("夏普比率", "+.2f", ""),
    ("最大回撤", ".2f", "%"),
    ("胜率", ".0f", "%"),
]:
    row = f"{metric:<18}"
    for k in keys:
        r = results[k]
        v = {"总收益率": r["total_ret"]*100,
             "夏普比率": r["sharpe"],
             "最大回撤": r["max_dd"]*100,
             "胜率": r["win_rate"]*100}[metric]
        row += f" {v:{fmt}}{suffix}".rjust(10)
    print(row)
print("-"*52)
for metric in ["n_open","n_rev","n_hedge","n_sl"]:
    label = {"n_open":"开仓","n_rev":"反手","n_hedge":"对冲","n_sl":"止损"}[metric]
    row = f"{label:<18}"
    for k in keys:
        row += f" {results[k][metric]}".rjust(10)
    print(row)
print("═"*52)
