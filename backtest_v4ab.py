# -*- coding: utf-8 -*-
"""
v4 三版本对比回测 (真实1分钟数据 3/19~3/30, 15分钟间隔)

V4_原版: 入场09:45 / 反手锚点=今开
V4a:    入场09:30 (昨收为基准直接入场) / 反手锚点=今开
V4b:    入场09:45 / 反手锚点=入场价（修复盲区）
"""

import sys, os, json, glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/tmp/etf-option-strategy")
from py_vollib.black import black

OUTPUT_DIR = "outputs"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 15分钟时间点（固定）──────────────────────────────────────
INTRADAY_TIMES = [
    "09:30","09:45","10:00","10:15","10:30","10:45",
    "11:00","11:15","11:30",
    "13:00","13:15","13:30","13:45","14:00","14:15",
    "14:30","14:45"
]
CLEAR_TIME = "14:45"
REV_CUTOFF = "13:30"

# ── v4 核心参数 ───────────────────────────────────────────────
RISK_FREE      = 0.018
MULTIPLIER     = 10000
COMMISSION     = 5.0
INIT_CAPITAL   = 200_000.0
OPEN_THRESH    = 0.005    # 开盘触发 >0.5%
REV_THRESH     = 0.008    # 反手阈值 0.8%
PROFIT_THRESH  = 0.05     # 对冲浮盈 5%
ROLL_DAYS      = 5
STOP_LOSS_PCT  = 0.35
HEDGE_CONFIRM_N= 2

MONTH_CN = {i: f"{i}月" for i in range(1,13)}
EXPIRY_SCHEDULE = {
    (2026, 3): date(2026, 3, 25), (2026, 4): date(2026, 4, 22),
    (2026, 5): date(2026, 5, 27), (2026, 6): date(2026, 6, 24),
    (2026, 9): date(2026, 9, 23),
}

# ── 三版本配置 ────────────────────────────────────────────────
VERSIONS = {
    "v4_orig": {
        "label":        "V4原版",
        "desc":         "入场09:45 / 反手锚点=今开",
        "entry_time":   "09:45",
        "rev_anchor":   "open",   # 反手锚点: 'open' or 'entry'
        "color":        "#94a3b8",
    },
    "v4a": {
        "label":        "V4a",
        "desc":         "入场09:30 / 反手锚点=今开",
        "entry_time":   "09:30",
        "rev_anchor":   "open",
        "color":        "#f59e0b",
    },
    "v4b": {
        "label":        "V4b",
        "desc":         "入场09:45 / 反手锚点=入场价",
        "entry_time":   "09:45",
        "rev_anchor":   "entry",
        "color":        "#3b82f6",
    },
}

# ── 基础工具函数 ──────────────────────────────────────────────
def get_expiry(y, m):
    return EXPIRY_SCHEDULE.get((y, m), date(y, m, 28))

def td_to_expiry(d, exp, ts):
    cnt = 0
    d = d
    while d < exp:
        d += timedelta(days=1)
        if d in ts: cnt += 1
    return cnt

def choose_expiry(d, ts):
    exp = get_expiry(d.year, d.month)
    if td_to_expiry(d, exp, ts) <= ROLL_DAYS:
        nm = d.month % 12 + 1
        ny = d.year + (1 if d.month == 12 else 0)
        exp = get_expiry(ny, nm)
    return exp

def est_iv(flag, K, F, T):
    base = 0.52 if flag == 'c' else 0.46
    adj  = (0.063 / max(T, 0.01)) ** 0.08
    skew = -0.15 * np.log(F / K)
    return max(0.15, min(base * adj + skew, 1.20))

def bk76(flag, F, K, T, r=RISK_FREE, sigma=0.35):
    try:
        return max(black(flag, F, K, max(T, 1/365), r, max(sigma, 0.05)), 0.0001)
    except:
        iv = max(0, F-K) if flag=='c' else max(0, K-F)
        return max(iv + F*sigma*np.sqrt(max(T,1/365))*0.15, 0.0001)

def T_yr(exp, dt):
    return max((datetime(exp.year,exp.month,exp.day,15,0)-dt).total_seconds()/(365*86400), 0.5/365)

class DCB:
    @staticmethod
    def build(cn, spot, exp):
        flag = 'c' if cn=="认购" else 'p'
        atm  = round(round(spot/0.05)*0.05, 3)
        return [{"code":  f"159915{cn[0]}{exp.strftime('%y%m')}{int((atm+i*0.05)*1000):05d}",
                 "name":  f"创业板ETF{cn[0]}{MONTH_CN[exp.month]}{int((atm+i*0.05)*1000)}",
                 "strike":round(atm+i*0.05,3), "expiry":exp.strftime("%Y-%m-%d"),
                 "type":cn, "flag":flag}
                for i in range(-4,5) if round(atm+i*0.05,3)>0]

    @staticmethod
    def atm(cn, spot, exp, dt):
        cs = DCB.build(cn, spot, exp)
        b  = min(cs, key=lambda c: abs(c["strike"]-spot))
        T  = T_yr(exp, dt); iv = est_iv(b["flag"], b["strike"], spot, T)
        b["iv"] = iv; b["premium"] = bk76(b["flag"], spot, b["strike"], T, RISK_FREE, iv)
        return b

    @staticmethod
    def otm(cn, spot, exp, dt, pct=0.03):
        cs    = DCB.build(cn, spot, exp)
        tgt   = spot*(1+pct) if cn=="认购" else spot*(1-pct)
        cands = [c for c in cs if (c["strike"]>spot if cn=="认购" else c["strike"]<spot)]
        if not cands: cands = cs
        b     = min(cands, key=lambda c: abs(c["strike"]-tgt))
        if b["strike"] == DCB.atm(cn, spot, exp, dt)["strike"]: return None
        T = T_yr(exp, dt); iv = est_iv(b["flag"], b["strike"], spot, T)
        b["iv"] = iv; b["premium"] = bk76(b["flag"], spot, b["strike"], T, RISK_FREE, iv)
        return b

class Pos:
    def __init__(self, c, prem, t, d="long"):
        self.contract=c; self.open_premium=prem
        self.current_price=prem; self.open_time=t; self.direction=d
    def pnl(self):
        return (1 if self.direction=="long" else -1)*(self.current_price-self.open_premium)*MULTIPLIER
    def pnl_pct(self):
        return (self.current_price-self.open_premium)/self.open_premium
    def reprice(self, spot, dt):
        exp = datetime.strptime(self.contract["expiry"],"%Y-%m-%d").date()
        T   = T_yr(exp, dt)
        iv  = est_iv(self.contract["flag"], self.contract["strike"], spot, T)
        self.current_price = bk76(self.contract["flag"], spot, self.contract["strike"], T, RISK_FREE, iv)


def get_prices(date_str, min_df, open_p):
    """从真实分钟数据提取15分钟时间点价格"""
    idx = min_df.index.tolist()
    out = []
    for ts in INTRADAY_TIMES:
        if ts == "09:30":
            out.append((ts, open_p))
        else:
            cands = [t for t in idx if t <= ts]
            out.append((ts, float(min_df.loc[max(cands), "close"]) if cands else open_p))
    return out


def run_version(ver_key: str, etf_df: pd.DataFrame, minute_data: dict):
    cfg         = VERSIONS[ver_key]
    entry_time  = cfg["entry_time"]
    rev_anchor  = cfg["rev_anchor"]  # 'open' or 'entry'
    trading_set = set(etf_df["date_obj"])
    capital     = INIT_CAPITAL
    daily_results, all_trades = [], []

    def rec(action, pos, ts, ds, reason, pnl=0.0, exp_info=""):
        all_trades.append({
            "date":ds, "time":ts, "action":action,
            "option_name":pos.contract["name"], "option_type":pos.contract["type"],
            "direction":pos.direction, "strike":pos.contract["strike"],
            "expiry":pos.contract["expiry"],
            "price":round(pos.current_price,6),
            "pnl":round(pnl,2), "reason":reason, "expiry_info":exp_info,
        })

    for idx in range(1, len(etf_df)):
        bar       = etf_df.iloc[idx]
        prev_bar  = etf_df.iloc[idx-1]
        date_str  = bar["date"]
        trade_dt  = bar["date_obj"]
        open_p    = bar["open"]
        prev_close= prev_bar["close"]

        expiry    = choose_expiry(trade_dt, trading_set)
        td_left   = td_to_expiry(trade_dt, expiry, trading_set)
        exp_label = f"{MONTH_CN[expiry.month]}合约(剩{td_left}日)"

        min_df = minute_data.get(date_str)
        if min_df is None:
            daily_results.append({"date":date_str,"etf_open":open_p,"etf_close":bar["close"],
                "etf_chg":round((bar["close"]-prev_close)/prev_close*100,3),
                "signal":"无","reversed":False,"hedged":False,"stop_loss":False,
                "daily_pnl":0.0,"daily_return":0.0,"total_value":round(capital,2),
                "expiry_label":exp_label})
            continue

        intraday = get_prices(date_str, min_df, open_p)

        positions: List[Pos] = []
        hedged = reversed_ = stop_loss_triggered = False
        hedge_confirm = 0
        entry_spot = None   # 记录实际入场时的ETF价格
        capital_at_open = capital

        # 开盘信号（全版本一致：昨收基准）
        change_rate = (open_p - prev_close) / prev_close
        signal = None
        if change_rate > OPEN_THRESH:    signal = "认购"
        elif change_rate < -OPEN_THRESH: signal = "认沽"

        for ts, spot in intraday[1:] if entry_time != "09:30" else intraday:
            # v4a 从09:30起遍历（含第一个点）；其余从第二个点起
            cur_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")
            for p in positions: p.reprice(spot, cur_dt)

            # 强平
            if ts >= CLEAR_TIME:
                for p in positions[:]:
                    pnl = p.pnl() - COMMISSION
                    act = "卖出平仓" if p.direction=="long" else "买入平仓"
                    rec(act, p, ts, date_str, f"{CLEAR_TIME}强平", pnl, exp_label)
                positions = []; break

            # ── 入场 ──
            if ts == entry_time and signal and not positions and not stop_loss_triggered:
                if entry_time == "09:30":
                    # v4a: 09:30直接入场，不做反转确认
                    open_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")
                    c = DCB.atm(signal, spot, expiry, open_dt)
                    pos = Pos(c, c["premium"], ts, "long")
                    positions.append(pos)
                    entry_spot = spot
                    rec("买入开仓", pos, ts, date_str,
                        f"09:30直接{'↑' if signal=='认购' else '↓'}{change_rate*100:.2f}%",
                        exp_info=exp_label)
                else:
                    # v4_orig / v4b: 09:45确认
                    cur_chg = (spot - open_p) / open_p
                    abandoned = (signal=="认购" and cur_chg < -OPEN_THRESH) or \
                                (signal=="认沽" and cur_chg >  OPEN_THRESH)
                    if abandoned:
                        signal = None
                    else:
                        open_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")
                        c = DCB.atm(signal, spot, expiry, open_dt)
                        pos = Pos(c, c["premium"], ts, "long")
                        positions.append(pos)
                        entry_spot = spot
                        rec("买入开仓", pos, ts, date_str,
                            f"09:45确认{'↑' if signal=='认购' else '↓'}{change_rate*100:.2f}%",
                            exp_info=exp_label)

            if not positions: continue

            # ── 止损 ──
            lp = next((p for p in positions if p.direction=="long"), None)
            if lp and not stop_loss_triggered:
                loss = (lp.open_premium - lp.current_price) / lp.open_premium
                if loss > STOP_LOSS_PCT:
                    rec("止损平仓", lp, ts, date_str,
                        f"止损:{loss*100:.1f}%", lp.pnl()-COMMISSION, exp_label)
                    positions.remove(lp)
                    sp = next((p for p in positions if p.direction=="short"), None)
                    if sp:
                        rec("买入平仓", sp, ts, date_str, "止损平对冲",
                            sp.pnl()-COMMISSION, exp_label)
                        positions.remove(sp)
                    stop_loss_triggered = True; hedge_confirm = 0; continue

            # ── 反手（截止13:30）──
            lp = next((p for p in positions if p.direction=="long"), None)
            if lp and not reversed_ and ts <= REV_CUTOFF:
                # 锚点选择
                anchor = entry_spot if (rev_anchor == "entry" and entry_spot) else open_p
                chg = (spot - anchor) / anchor
                rev_need = (lp.contract["flag"]=='c' and chg < -REV_THRESH) or \
                           (lp.contract["flag"]=='p' and chg >  REV_THRESH)
                if rev_need:
                    pnl_c = lp.pnl() - COMMISSION
                    anchor_label = "入场价" if rev_anchor=="entry" else "今开"
                    rec("反手平仓", lp, ts, date_str,
                        f"相对{anchor_label}偏{chg*100:.2f}%反手", pnl_c, exp_label)
                    positions.remove(lp)
                    new_cn = "认沽" if lp.contract["flag"]=='c' else "认购"
                    nc = DCB.atm(new_cn, spot, expiry, cur_dt)
                    np_ = Pos(nc, nc["premium"], ts, "long")
                    positions.append(np_)
                    entry_spot = spot   # 反手后更新入场价
                    rec("反手开仓", np_, ts, date_str, f"反手→{new_cn}", exp_info=exp_label)
                    reversed_ = True; hedge_confirm = 0

            # ── 对冲（连续2次确认）──
            if not hedged:
                lp = next((p for p in positions if p.direction=="long"), None)
                if lp and lp.pnl_pct() > PROFIT_THRESH:
                    hedge_confirm += 1
                    if hedge_confirm >= HEDGE_CONFIRM_N:
                        hc = DCB.otm(lp.contract["type"], spot, expiry, cur_dt, 0.03)
                        if hc:
                            hp = Pos(hc, hc["premium"], ts, "short")
                            positions.append(hp)
                            rec("卖出对冲", hp, ts, date_str,
                                f"连续{hedge_confirm}次浮盈→对冲", pnl=0, exp_info=exp_label)
                            hedged = True
                else:
                    hedge_confirm = 0

        day_pnl = sum(t["pnl"] for t in all_trades if t["date"]==date_str and t["pnl"]!=0)
        capital += day_pnl
        ret = day_pnl / capital_at_open if capital_at_open else 0.0
        daily_results.append({
            "date":date_str, "etf_open":open_p, "etf_close":bar["close"],
            "etf_chg":round((bar["close"]-prev_close)/prev_close*100,3),
            "signal":signal or "无", "reversed":reversed_,
            "hedged":hedged, "stop_loss":stop_loss_triggered,
            "daily_pnl":round(day_pnl,2), "daily_return":round(ret,6),
            "total_value":round(capital,2), "expiry_label":exp_label,
        })

    daily_df  = pd.DataFrame(daily_results)
    trades_df = pd.DataFrame(all_trades)
    total_ret = (capital - INIT_CAPITAL) / INIT_CAPITAL
    n_days    = len(daily_df)
    rets      = daily_df["daily_return"]
    sharpe    = (rets.mean()-RISK_FREE/250)/rets.std()*np.sqrt(250) if rets.std()>0 else 0
    cum       = (1+rets).cumprod()
    max_dd    = ((cum-cum.cummax())/cum.cummax()).min()
    win_rate  = (daily_df["daily_pnl"]>0).mean()
    avg_w = daily_df[daily_df["daily_pnl"]>0]["daily_pnl"].mean() if (daily_df["daily_pnl"]>0).any() else 0
    avg_l = abs(daily_df[daily_df["daily_pnl"]<0]["daily_pnl"].mean()) if (daily_df["daily_pnl"]<0).any() else 1
    plr   = avg_w / avg_l
    return {
        "label":     cfg["label"],
        "desc":      cfg["desc"],
        "color":     cfg["color"],
        "daily_df":  daily_df,
        "trades_df": trades_df,
        "final_cap": capital,
        "total_ret": total_ret,
        "sharpe":    sharpe,
        "max_dd":    max_dd,
        "win_rate":  win_rate,
        "plr":       plr,
        "n_open":    int((trades_df["action"]=="买入开仓").sum()) if len(trades_df) else 0,
        "n_rev":     int((trades_df["action"]=="反手开仓").sum()) if len(trades_df) else 0,
        "n_hedge":   int((trades_df["action"]=="卖出对冲").sum()) if len(trades_df) else 0,
        "n_sl":      int((trades_df["action"]=="止损平仓").sum()) if len(trades_df) else 0,
        "n_days":    n_days,
    }


# ── 数据加载 ──────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "159915_*_1min.csv")))
date_strs = [f"{os.path.basename(f).split('_')[1][:4]}-"
             f"{os.path.basename(f).split('_')[1][4:6]}-"
             f"{os.path.basename(f).split('_')[1][6:8]}" for f in files]
print(f"数据: {len(date_strs)} 个交易日  {date_strs[0]} ~ {date_strs[-1]}")

def build_daily(ds_list):
    rows = []
    for ds in ds_list:
        f = os.path.join(UPLOAD_DIR, f"159915_{ds.replace('-','')}_1min.csv")
        if not os.path.exists(f): continue
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        rows.append({"date":ds,"date_obj":date.fromisoformat(ds),
                     "open":df["open"].iloc[0],"high":df["high"].max(),
                     "low":df["low"].min(),"close":df["close"].iloc[-1]})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

etf_df = build_daily(date_strs)

minute_data = {}
for ds in date_strs:
    f = os.path.join(UPLOAD_DIR, f"159915_{ds.replace('-','')}_1min.csv")
    df = pd.read_csv(f, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["time_str"] = pd.to_datetime(df["day"]).dt.strftime("%H:%M")
    minute_data[ds] = df.set_index("time_str")

# ── 运行三版本 ────────────────────────────────────────────────
results = {}
for key in VERSIONS:
    cfg = VERSIONS[key]
    print(f"\n{'='*54}")
    print(f"  [{cfg['label']}] {cfg['desc']}")
    r = run_version(key, etf_df, minute_data)
    results[key] = r
    print(f"  总收益:{r['total_ret']*100:+.2f}%  夏普:{r['sharpe']:+.2f}  "
          f"回撤:{r['max_dd']*100:.2f}%  胜率:{r['win_rate']*100:.0f}%")
    print(f"  盈亏比:{r['plr']:.2f}  "
          f"开仓:{r['n_open']} 反手:{r['n_rev']} 对冲:{r['n_hedge']} 止损:{r['n_sl']}")

# ── 打印对比表 ────────────────────────────────────────────────
print(f"\n{'═'*54}")
print("       三版本汇总对比（15分钟间隔，真实数据）")
print(f"{'═'*54}")
print(f"{'指标':<16} {'V4原版':>12} {'V4a(09:30)':>12} {'V4b(入场锚)':>12}")
print("-"*54)
rows_summary = [
    ("总收益率", "total_ret", lambda v: f"{v*100:+.2f}%"),
    ("夏普比率", "sharpe",    lambda v: f"{v:+.2f}"),
    ("最大回撤", "max_dd",    lambda v: f"{v*100:.2f}%"),
    ("胜率",     "win_rate",  lambda v: f"{v*100:.0f}%"),
    ("盈亏比",   "plr",       lambda v: f"{v:.3f}"),
    ("开仓次数", "n_open",    lambda v: str(v)),
    ("反手次数", "n_rev",     lambda v: str(v)),
    ("对冲次数", "n_hedge",   lambda v: str(v)),
    ("止损次数", "n_sl",      lambda v: str(v)),
]
for label, key, fmt in rows_summary:
    row = f"{label:<16}"
    for vk in ["v4_orig","v4a","v4b"]:
        row += fmt(results[vk][key]).rjust(12)
    print(row)
print("═"*54)

# ── 每日明细对比 ──────────────────────────────────────────────
print(f"\n{'─'*80}")
print(f"{'日期':<12}  {'ETF涨跌':>7}  "
      f"{'V4_盈亏':>9} {'操作':<6}  "
      f"{'V4a_盈亏':>9} {'操作':<6}  "
      f"{'V4b_盈亏':>9} {'操作':<6}")
print("─"*80)
for _, row in results["v4_orig"]["daily_df"].iterrows():
    ds = row["date"]
    chg_str = f"{row['etf_chg']:+.2f}%"
    def day_str(vk):
        df = results[vk]["daily_df"]
        r = df[df["date"]==ds]
        if r.empty: return "       -", "  -   "
        r = r.iloc[0]
        flags = ("R" if r["reversed"] else "") + ("H" if r["hedged"] else "") + ("S" if r["stop_loss"] else "")
        return f"{r['daily_pnl']:>+9.2f}", f"  {flags if flags else '-':<4}"
    p0,f0 = day_str("v4_orig")
    pa,fa = day_str("v4a")
    pb,fb = day_str("v4b")
    print(f"{ds:<12}  {chg_str:>7}  {p0}{f0}  {pa}{fa}  {pb}{fb}")
print("─"*80)

# ── 保存JSON ──────────────────────────────────────────────────
summary = {k: {kk: vv for kk,vv in v.items() if kk not in ("daily_df","trades_df")}
           for k,v in results.items()}
with open(os.path.join(OUTPUT_DIR,"backtest_v4ab_perf.json"),"w",encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ── HTML报告 ──────────────────────────────────────────────────
print("\n生成HTML报告...")

keys   = ["v4_orig","v4a","v4b"]
dates  = results["v4_orig"]["daily_df"]["date"].tolist()
colors = {k: results[k]["color"] for k in keys}
labels = {k: results[k]["label"] for k in keys}
descs  = {k: results[k]["desc"]  for k in keys}

nav_js = {k: json.dumps([round(v/INIT_CAPITAL,6) for v in results[k]["daily_df"]["total_value"]]) for k in keys}
pnl_js = {k: json.dumps([round(v,2) for v in results[k]["daily_df"]["daily_pnl"]]) for k in keys}

def metric_cards():
    cards = ""
    for k in keys:
        r = results[k]
        tr_cls = "pos" if r["total_ret"]>=0 else "neg"
        sh_cls = "pos" if r["sharpe"]>=0 else "neg"
        cards += f"""
      <div class="card ver-card">
        <div class="ver-header" style="border-color:{r['color']}">
          <span class="dot" style="background:{r['color']}"></span>
          <strong>{r['label']}</strong>
          <small>{r['desc']}</small>
        </div>
        <div class="metric-grid">
          <div class="m"><span class="{tr_cls} big">{r['total_ret']*100:+.2f}%</span><br><small>总收益</small></div>
          <div class="m"><span class="{sh_cls} big">{r['sharpe']:+.2f}</span><br><small>夏普</small></div>
          <div class="m"><span class="neg big">{r['max_dd']*100:.2f}%</span><br><small>最大回撤</small></div>
          <div class="m"><span class="big">{r['win_rate']*100:.0f}%</span><br><small>胜率</small></div>
          <div class="m"><span class="big">{r['plr']:.3f}</span><br><small>盈亏比</small></div>
          <div class="m"><span class="neu big">{r['n_open']}/{r['n_rev']}/{r['n_hedge']}/{r['n_sl']}</span><br><small>开/反/冲/损</small></div>
        </div>
      </div>"""
    return cards

def daily_compare_table():
    rows = ""
    for ds in dates:
        row = f"<td>{ds}</td>"
        # ETF涨跌
        r0 = results["v4_orig"]["daily_df"]
        etf_row = r0[r0["date"]==ds]
        if not etf_row.empty:
            chg = etf_row.iloc[0]["etf_chg"]
            chg_c = "#22c55e" if chg>0 else "#ef4444"
            row += f"<td style='color:{chg_c}'>{chg:+.2f}%</td>"
            row += f"<td>{etf_row.iloc[0]['etf_open']:.3f}</td>"
            row += f"<td>{etf_row.iloc[0]['etf_close']:.3f}</td>"
        else:
            row += "<td colspan=3>-</td>"
        for k in keys:
            df = results[k]["daily_df"]
            dr = df[df["date"]==ds]
            if dr.empty:
                row += "<td colspan=3>-</td>"
                continue
            r = dr.iloc[0]
            pc = "#22c55e" if r["daily_pnl"]>0 else "#ef4444" if r["daily_pnl"]<0 else "#94a3b8"
            flags = ("R" if r.get("reversed") else "") + ("H" if r.get("hedged") else "") + ("S" if r.get("stop_loss") else "")
            row += (f"<td style='color:{pc}'>{r['daily_pnl']:+.2f}</td>"
                    f"<td>{flags or '-'}</td>"
                    f"<td style='color:{pc}'>{r['total_value']:,.2f}</td>")
        rows += f"<tr>{row}</tr>\n"
    return rows

def trades_tab(k):
    tdf = results[k]["trades_df"]
    if len(tdf)==0:
        return "<tr><td colspan=8 style='text-align:center;color:#64748b'>无交易记录</td></tr>"
    out = ""
    for _, t in tdf.iterrows():
        pc = "#22c55e" if t["pnl"]>0 else "#ef4444" if t["pnl"]<0 else "#94a3b8"
        pnl_str = f"<span style='color:{pc}'>{t['pnl']:+.2f}</span>" if t["pnl"]!=0 else "-"
        act_cls = "buy" if "买" in t["action"] else "sell"
        out += (f"<tr><td>{t['date']}</td><td>{t['time']}</td>"
                f"<td><span class='badge {act_cls}'>{t['action']}</span></td>"
                f"<td>{t['option_name']}</td><td>{t['strike']}</td>"
                f"<td>{t['price']:.4f}</td><td>{pnl_str}</td>"
                f"<td style='color:#94a3b8;font-size:11px'>{t.get('reason','')}</td></tr>")
    return out

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>v4 三版本对比: 入场时机 vs 反手锚点</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0f172a;color:#e2e8f0;font-size:13px;line-height:1.6}}
.container{{max-width:1440px;margin:0 auto;padding:24px}}
h1{{font-size:21px;font-weight:700;color:#f1f5f9;margin-bottom:4px}}
.sub{{color:#64748b;font-size:12px;margin-bottom:20px}}
.grid{{display:grid;gap:14px}}
.g3{{grid-template-columns:repeat(3,1fr)}}
.g2{{grid-template-columns:repeat(2,1fr)}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px}}
.ver-card .ver-header{{border-left:3px solid;padding:6px 10px;margin-bottom:12px;
                       display:flex;align-items:center;gap:8px;border-radius:4px;
                       background:#0f172a}}
.ver-header strong{{font-size:15px;color:#f1f5f9}}
.ver-header small{{color:#64748b;font-size:11px}}
.metric-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;text-align:center}}
.m{{padding:8px 4px}}
.big{{font-size:18px;font-weight:700}}
.pos{{color:#22c55e}}.neg{{color:#ef4444}}.neu{{color:#94a3b8}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;flex-shrink:0}}
.chart-wrap{{position:relative;height:220px}}
.card h2{{font-size:13px;color:#94a3b8;margin-bottom:12px;font-weight:500}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{color:#64748b;font-weight:500;padding:7px 5px;border-bottom:1px solid #334155;text-align:center}}
td{{padding:6px 5px;border-bottom:1px solid #1e293b;text-align:center}}
tr:hover td{{background:#334155}}
.badge{{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;
        background:#3b4a6b;color:#93c5fd}}
.badge.buy{{background:#14532d;color:#4ade80}}
.badge.sell{{background:#450a0a;color:#f87171}}
.tabs{{display:flex;gap:4px;margin-bottom:12px}}
.tab{{padding:5px 14px;border-radius:6px;cursor:pointer;font-size:12px;
      background:#334155;color:#94a3b8;border:none}}
.tab.active{{background:#3b82f6;color:#fff}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
.diff-badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px;
             background:#1e3a5f;color:#93c5fd;margin-left:6px}}
.insight{{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px 16px;
          font-size:12px;line-height:1.8;color:#94a3b8}}
.insight b{{color:#e2e8f0}}
</style>
</head>
<body>
<div class="container">
  <h1>v4策略 三版本对比：入场时机 vs 反手锚点</h1>
  <div class="sub">
    回测周期: {dates[0]} ~ {dates[-1]} (7交易日, 真实1分钟数据, 15分钟检查间隔)
    &nbsp;|&nbsp; V4原版 / V4a(09:30入场) / V4b(入场价反手锚点)
  </div>

  <!-- 设计差异说明 -->
  <div class="insight" style="margin-bottom:16px">
    <b>三版本关键差异：</b><br>
    <span style="color:{colors['v4_orig']}">▎</span> <b>V4原版</b>：入场时间=09:45（等15分钟确认），反手锚点=今天开盘价 &nbsp;|&nbsp;
    <span style="color:{colors['v4a']}">▎</span> <b>V4a</b>：入场时间=<b>09:30</b>（昨收信号直接入场，无延迟），反手锚点=今天开盘价 &nbsp;|&nbsp;
    <span style="color:{colors['v4b']}">▎</span> <b>V4b</b>：入场时间=09:45，反手锚点=<b>实际入场价</b>（消除盲区，偏离入场价-0.8%即反手）
  </div>

  <!-- 核心指标卡 -->
  <div class="grid g3" style="margin-bottom:16px">
    {metric_cards()}
  </div>

  <!-- 净值曲线 -->
  <div class="card" style="margin-bottom:14px">
    <h2>净值曲线对比</h2>
    <div class="chart-wrap" style="height:250px"><canvas id="navChart"></canvas></div>
  </div>

  <!-- 每日盈亏 -->
  <div class="card" style="margin-bottom:14px">
    <h2>每日盈亏对比 (元)</h2>
    <div class="chart-wrap"><canvas id="pnlChart"></canvas></div>
  </div>

  <!-- 每日明细对比表 -->
  <div class="card" style="margin-bottom:14px">
    <h2>每日明细对比（R=反手 H=对冲 S=止损）</h2>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th rowspan="2">日期</th>
          <th colspan="3">ETF</th>
          {"".join(f'<th colspan="3" style="border-left:1px solid #475569"><span class="dot" style="background:{colors[k]};margin-right:4px"></span>{labels[k]}</th>' for k in keys)}
        </tr>
        <tr>
          <th>涨跌</th><th>开</th><th>收</th>
          {"".join('<th style="border-left:1px solid #475569">盈亏</th><th>操作</th><th>净值</th>' for _ in keys)}
        </tr>
      </thead>
      <tbody>{daily_compare_table()}</tbody>
    </table>
    </div>
  </div>

  <!-- 交易记录 -->
  <div class="card">
    <h2>交易记录详情</h2>
    <div class="tabs">
      <button class="tab active" onclick="showTab('tv4')">V4原版</button>
      <button class="tab" onclick="showTab('tv4a')">V4a (09:30入场)</button>
      <button class="tab" onclick="showTab('tv4b')">V4b (入场价锚点)</button>
    </div>
    <div id="tv4" class="tab-content active">
      <div style="overflow-x:auto;max-height:340px;overflow-y:auto">
      <table>
        <thead><tr><th>日期</th><th>时间</th><th>操作</th><th>合约</th><th>行权价</th><th>权利金</th><th>盈亏</th><th>原因</th></tr></thead>
        <tbody>{trades_tab('v4_orig')}</tbody>
      </table></div>
    </div>
    <div id="tv4a" class="tab-content">
      <div style="overflow-x:auto;max-height:340px;overflow-y:auto">
      <table>
        <thead><tr><th>日期</th><th>时间</th><th>操作</th><th>合约</th><th>行权价</th><th>权利金</th><th>盈亏</th><th>原因</th></tr></thead>
        <tbody>{trades_tab('v4a')}</tbody>
      </table></div>
    </div>
    <div id="tv4b" class="tab-content">
      <div style="overflow-x:auto;max-height:340px;overflow-y:auto">
      <table>
        <thead><tr><th>日期</th><th>时间</th><th>操作</th><th>合约</th><th>行权价</th><th>权利金</th><th>盈亏</th><th>原因</th></tr></thead>
        <tbody>{trades_tab('v4b')}</tbody>
      </table></div>
    </div>
  </div>
</div>

<script>
const dates = {json.dumps(dates)};
const navs = {{"v4_orig":{nav_js['v4_orig']},"v4a":{nav_js['v4a']},"v4b":{nav_js['v4b']}}};
const pnls = {{"v4_orig":{pnl_js['v4_orig']},"v4a":{pnl_js['v4a']},"v4b":{pnl_js['v4b']}}};
const cols = {{"v4_orig":"{colors['v4_orig']}","v4a":"{colors['v4a']}","v4b":"{colors['v4b']}"}};
const lbls = {{"v4_orig":"V4原版","v4a":"V4a(09:30)","v4b":"V4b(入场锚)"}};
const axOpts = {{
  x:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:10}}}}}},
  y:{{grid:{{color:'#334155'}},ticks:{{color:'#64748b',font:{{size:10}}}}}}
}};
new Chart(document.getElementById('navChart'),{{
  type:'line',
  data:{{labels:dates,datasets:Object.keys(navs).map(k=>({{
    label:lbls[k],data:navs[k],borderColor:cols[k],
    backgroundColor:cols[k]+'20',fill:true,tension:0.35,
    pointRadius:4,borderWidth:2.5,pointBackgroundColor:cols[k]
  }})) }},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:true,labels:{{color:'#94a3b8',font:{{size:11}}}}}}}},
    scales:{{...axOpts,y:{{...axOpts.y,ticks:{{...axOpts.y.ticks,callback:v=>(v*100).toFixed(2)+'%'}}}}}}
  }}
}});
new Chart(document.getElementById('pnlChart'),{{
  type:'bar',
  data:{{labels:dates,datasets:Object.keys(pnls).map(k=>({{
    label:lbls[k],data:pnls[k],backgroundColor:cols[k]+'88',borderColor:cols[k],borderWidth:1
  }})) }},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:true,labels:{{color:'#94a3b8',font:{{size:11}}}}}}}},
    scales:axOpts
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

rpt = os.path.join(OUTPUT_DIR, "backtest_v4ab.html")
with open(rpt, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML报告: {rpt}")
print("完成！")
