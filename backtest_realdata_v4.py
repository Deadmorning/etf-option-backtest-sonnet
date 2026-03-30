# -*- coding: utf-8 -*-
"""
真实分钟数据回测 v4 (3/19 ~ 3/30)
使用上传的 159915 1分钟真实数据，按v4策略逻辑进行回测
"""

import sys, os, json, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/tmp/etf-option-strategy")
from py_vollib.black import black

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "backtest_realdata_v4.log"),
                            encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bt_real_v4")

# ── 全局参数（与v4一致）────────────────────────────────────────
RISK_FREE      = 0.018
MULTIPLIER     = 10000
COMMISSION     = 5.0
INIT_CAPITAL   = 200_000.0
OPEN_THRESH    = 0.005
REV_THRESH     = 0.008
PROFIT_THRESH  = 0.05
ROLL_DAYS      = 5
ENTRY_TIME     = "09:45"
STOP_LOSS_PCT  = 0.35
HEDGE_CONFIRM_N= 2

# 15分钟间隔时间点
INTRADAY_TIMES = ["09:30","09:45","10:00","10:15","10:30","10:45",
                  "11:00","11:15","11:30",
                  "13:00","13:15","13:30","13:45","14:00","14:15",
                  "14:30","14:45"]
CLEAR_TIME = "14:45"

MONTH_CN = {1:"1月",2:"2月",3:"3月",4:"4月",5:"5月",
            6:"6月",7:"7月",8:"8月",9:"9月",10:"10月",
            11:"11月",12:"12月"}

EXPIRY_SCHEDULE = {
    (2026,  1): date(2026,  1, 28),
    (2026,  2): date(2026,  2, 25),
    (2026,  3): date(2026,  3, 25),
    (2026,  4): date(2026,  4, 22),
    (2026,  5): date(2026,  5, 27),
    (2026,  6): date(2026,  6, 24),
    (2026,  9): date(2026,  9, 23),
    (2026, 12): date(2026, 12, 23),
}

def get_expiry(year, month):
    return EXPIRY_SCHEDULE.get((year, month), date(year, month, 28))

def trading_days_to_expiry(trade_date, expiry, trading_set):
    d, cnt = trade_date, 0
    while d < expiry:
        d += timedelta(days=1)
        if d in trading_set:
            cnt += 1
    return cnt

def choose_expiry(trade_date, trading_set):
    curr_exp = get_expiry(trade_date.year, trade_date.month)
    td = trading_days_to_expiry(trade_date, curr_exp, trading_set)
    if td <= ROLL_DAYS:
        nm = trade_date.month + 1 if trade_date.month < 12 else 1
        ny = trade_date.year if trade_date.month < 12 else trade_date.year + 1
        return get_expiry(ny, nm)
    return curr_exp

SVI_BASE_IV = {"call": 0.52, "put": 0.46}

def estimate_iv(flag, K, F, T):
    base = SVI_BASE_IV["call"] if flag == 'c' else SVI_BASE_IV["put"]
    T_ref = 0.063
    term_adj = (T_ref / max(T, 0.01)) ** 0.08
    moneyness = np.log(F / K)
    skew = -0.15 * moneyness
    iv = base * term_adj + skew
    return max(0.15, min(iv, 1.20))

def bk76(flag, F, K, T, r=RISK_FREE, sigma=0.35):
    try:
        p = black(flag, F, K, max(T, 1/365), r, max(sigma, 0.05))
        return max(p, 0.0001)
    except Exception:
        iv = max(0, F-K) if flag=='c' else max(0, K-F)
        return max(iv + F*sigma*np.sqrt(max(T,1/365))*0.15, 0.0001)

def T_years(expiry, current_dt):
    exp_dt = datetime(expiry.year, expiry.month, expiry.day, 15, 0)
    delta  = exp_dt - current_dt
    return max(delta.total_seconds() / (365 * 86400), 0.5/365)

class DynamicContractBuilder:
    @staticmethod
    def build(cn_type, spot, expiry):
        flag   = 'c' if cn_type == "认购" else 'p'
        month  = MONTH_CN[expiry.month]
        atm_k  = round(round(spot / 0.05) * 0.05, 3)
        strikes = [round(atm_k + i*0.05, 3) for i in range(-4, 5)]
        contracts = []
        for K in strikes:
            if K <= 0: continue
            code = f"159915{cn_type[0]}{expiry.strftime('%y%m')}{int(K*1000):05d}"
            name = f"创业板ETF{cn_type[0]}{month}{int(K*1000)}"
            contracts.append({"code": code, "name": name,
                               "strike": K, "expiry": expiry.strftime("%Y-%m-%d"),
                               "type": cn_type, "flag": flag})
        return contracts

    @staticmethod
    def get_atm(cn_type, spot, expiry, current_dt):
        contracts = DynamicContractBuilder.build(cn_type, spot, expiry)
        best = min(contracts, key=lambda c: abs(c["strike"] - spot))
        T = T_years(expiry, current_dt)
        iv = estimate_iv(best["flag"], best["strike"], spot, T)
        best["iv"] = iv
        best["premium"] = bk76(best["flag"], spot, best["strike"], T, RISK_FREE, iv)
        return best

    @staticmethod
    def get_otm(cn_type, spot, expiry, current_dt, otm_pct=0.03):
        contracts = DynamicContractBuilder.build(cn_type, spot, expiry)
        if cn_type == "认购":
            target = spot * (1 + otm_pct)
            cands  = [c for c in contracts if c["strike"] > spot]
        else:
            target = spot * (1 - otm_pct)
            cands  = [c for c in contracts if c["strike"] < spot]
        if not cands: cands = contracts
        best = min(cands, key=lambda c: abs(c["strike"] - target))
        atm_k = DynamicContractBuilder.get_atm(cn_type, spot, expiry, current_dt)["strike"]
        if best["strike"] == atm_k: return None
        T = T_years(expiry, current_dt)
        iv = estimate_iv(best["flag"], best["strike"], spot, T)
        best["iv"] = iv
        best["premium"] = bk76(best["flag"], spot, best["strike"], T, RISK_FREE, iv)
        return best

class Position:
    def __init__(self, contract, open_premium, open_time, direction="long"):
        self.contract      = contract
        self.open_premium  = open_premium
        self.current_price = open_premium
        self.open_time     = open_time
        self.direction     = direction
        self.qty           = 1

    def pnl(self):
        sign = 1 if self.direction == "long" else -1
        return sign * (self.current_price - self.open_premium) * MULTIPLIER

    def pnl_pct(self):
        return (self.current_price - self.open_premium) / self.open_premium

    def reprice(self, spot, current_dt):
        exp = datetime.strptime(self.contract["expiry"], "%Y-%m-%d").date()
        T   = T_years(exp, current_dt)
        iv  = estimate_iv(self.contract["flag"], self.contract["strike"], spot, T)
        self.current_price = bk76(self.contract["flag"], spot,
                                  self.contract["strike"], T, RISK_FREE, iv)


# ── 加载真实1分钟数据 ──────────────────────────────────────────
UPLOAD_DIR = "uploads"

def load_minute_data(date_str: str) -> Optional[pd.DataFrame]:
    """加载指定日期的1分钟数据，返回时间→close价格的字典"""
    fname = os.path.join(UPLOAD_DIR, f"159915_{date_str.replace('-','')}_1min.csv")
    if not os.path.exists(fname):
        return None
    df = pd.read_csv(fname, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["time_str"] = pd.to_datetime(df["day"]).dt.strftime("%H:%M")
    df = df.set_index("time_str")
    return df

def get_real_intraday(date_str: str, min_df: pd.DataFrame,
                      day_open: float) -> List[Tuple[str, float]]:
    """
    从1分钟数据中提取15分钟时间点的价格。
    - 09:30 → 用当天第一根K线开盘价
    - 其余时间点 → 找最近一根<=该时间的K线收盘价
    """
    result = []
    time_index = min_df.index.tolist()

    for ts in INTRADAY_TIMES:
        if ts == "09:30":
            # 用第一根K线(09:31)的open
            price = day_open
        else:
            # 找 <= ts 的最近K线收盘价
            candidates = [t for t in time_index if t <= ts]
            if candidates:
                nearest = max(candidates)
                price = min_df.loc[nearest, "close"]
            else:
                price = day_open
        result.append((ts, float(price)))
    return result


# ── 主回测引擎（真实数据版）─────────────────────────────────────
class RealDataBacktesterV4:
    def __init__(self, etf_df: pd.DataFrame, minute_data: dict):
        self.etf          = etf_df.reset_index(drop=True)
        self.trading_set  = set(etf_df["date_obj"])
        self.minute_data  = minute_data   # {date_str: pd.DataFrame}
        self.capital      = INIT_CAPITAL
        self.daily_results: List[dict] = []
        self.all_trades:    List[dict] = []

    def _trade(self, action, pos, ts, date_str, reason, pnl=0.0, expiry_info=""):
        self.all_trades.append({
            "date":          date_str,
            "time":          ts,
            "action":        action,
            "option_code":   pos.contract["code"],
            "option_name":   pos.contract["name"],
            "option_type":   pos.contract["type"],
            "direction":     pos.direction,
            "strike":        pos.contract["strike"],
            "expiry":        pos.contract["expiry"],
            "price":         round(pos.current_price, 6),
            "premium_value": round(pos.current_price * MULTIPLIER, 2),
            "commission":    COMMISSION,
            "pnl":           round(pnl, 2),
            "reason":        reason,
            "expiry_info":   expiry_info,
        })

    def run_day(self, idx: int) -> dict:
        bar       = self.etf.iloc[idx]
        prev_bar  = self.etf.iloc[idx - 1]
        date_str  = bar["date"]
        trade_dt  = bar["date_obj"]
        open_p    = bar["open"]
        prev_close= prev_bar["close"]

        expiry    = choose_expiry(trade_dt, self.trading_set)
        td_left   = trading_days_to_expiry(trade_dt, expiry, self.trading_set)
        exp_label = f"{MONTH_CN[expiry.month]}合约(剩{td_left}交易日)"

        # 使用真实分钟数据（若有），否则fallback到模拟
        min_df = self.minute_data.get(date_str)
        data_source = "真实分钟"
        if min_df is not None:
            intraday = get_real_intraday(date_str, min_df, open_p)
        else:
            # fallback: 合成路径（不应在当前数据集中出现）
            from step5_backtest_v4 import gen_intraday
            intraday = gen_intraday(
                {"open": open_p, "high": bar["high"],
                 "low": bar["low"], "close": bar["close"]},
                seed_offset=idx
            )
            data_source = "模拟路径"

        positions: List[Position] = []
        hedged = reversed_ = stop_loss_triggered = False
        hedge_confirm = 0
        capital_at_open = self.capital

        # 开盘信号
        change_rate = (open_p - prev_close) / prev_close
        signal = None
        if change_rate > OPEN_THRESH:
            signal = "认购"
        elif change_rate < -OPEN_THRESH:
            signal = "认沽"

        # 记录日内明细（用于报告）
        intraday_detail = []

        for ts, spot in intraday[1:]:
            cur_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")
            for p in positions:
                p.reprice(spot, cur_dt)

            # 记录明细
            lp_price = next((p.current_price for p in positions if p.direction=="long"), None)
            intraday_detail.append({
                "time": ts,
                "spot": round(spot, 4),
                "option_price": round(lp_price, 6) if lp_price else None,
                "positions": len(positions)
            })

            # 14:45 强平
            if ts >= CLEAR_TIME:
                for p in positions[:]:
                    pnl = p.pnl() - COMMISSION
                    act = "卖出平仓" if p.direction == "long" else "买入平仓"
                    self._trade(act, p, ts, date_str, "14:45强平", pnl,
                                expiry_info=exp_label)
                    logger.info(f"{date_str} {ts} 【清仓】{p.contract['name']} "
                                f"P={p.current_price:.4f} 盈亏={pnl:.2f}")
                positions = []
                break

            # 09:45 入场确认
            if ts == ENTRY_TIME and signal and not positions and not stop_loss_triggered:
                cur_chg = (spot - open_p) / open_p
                if signal == "认购" and cur_chg < -OPEN_THRESH:
                    logger.info(f"{date_str} {ts} 【放弃入场】开盘↑信号已反转({cur_chg*100:.2f}%)")
                    signal = None
                elif signal == "认沽" and cur_chg > OPEN_THRESH:
                    logger.info(f"{date_str} {ts} 【放弃入场】开盘↓信号已反转({cur_chg*100:.2f}%)")
                    signal = None
                else:
                    open_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")
                    c = DynamicContractBuilder.get_atm(signal, spot, expiry, open_dt)
                    pos = Position(c, c["premium"], ts, "long")
                    positions.append(pos)
                    self._trade("买入开仓", pos, ts, date_str,
                                f"09:45确认{'↑' if signal=='认购' else '↓'}{change_rate*100:.2f}%",
                                expiry_info=exp_label)
                    logger.info(f"{date_str} {ts} 【开仓】{c['name']} K={c['strike']} "
                                f"P={c['premium']:.4f} T={T_years(expiry,open_dt)*365:.0f}天 "
                                f"到期={expiry} σ={c['iv']:.2f} [{data_source}]")

            if not positions:
                continue

            # 止损检查
            lp = next((p for p in positions if p.direction == "long"), None)
            if lp and not stop_loss_triggered:
                loss_pct = (lp.open_premium - lp.current_price) / lp.open_premium
                if loss_pct > STOP_LOSS_PCT:
                    pnl_sl = lp.pnl() - COMMISSION
                    self._trade("止损平仓", lp, ts, date_str,
                                f"止损:{loss_pct*100:.1f}%跌幅>{STOP_LOSS_PCT*100:.0f}%",
                                pnl_sl, expiry_info=exp_label)
                    logger.info(f"{date_str} {ts} 【止损】{lp.contract['name']} "
                                f"跌幅={loss_pct*100:.1f}% 亏损={pnl_sl:.2f}")
                    positions.remove(lp)
                    sp = next((p for p in positions if p.direction == "short"), None)
                    if sp:
                        pnl_sp = sp.pnl() - COMMISSION
                        self._trade("买入平仓", sp, ts, date_str,
                                    "止损同步平对冲仓", pnl_sp, expiry_info=exp_label)
                        positions.remove(sp)
                    stop_loss_triggered = True
                    hedge_confirm = 0
                    continue

            # 反手验证（阈值0.8%，截止13:30）
            lp = next((p for p in positions if p.direction == "long"), None)
            if lp and not reversed_ and ts <= "13:30":
                chg = (spot - open_p) / open_p
                rev_needed = (lp.contract["flag"] == 'c' and chg < -REV_THRESH) or \
                             (lp.contract["flag"] == 'p' and chg >  REV_THRESH)
                if rev_needed:
                    pnl_c = lp.pnl() - COMMISSION
                    self._trade("反手平仓", lp, ts, date_str,
                                f"偏离{chg*100:.2f}%反手", pnl_c, expiry_info=exp_label)
                    logger.info(f"{date_str} {ts} 【反手平仓】{lp.contract['name']} "
                                f"盈亏={pnl_c:.2f}")
                    positions.remove(lp)
                    new_cn = "认沽" if lp.contract["flag"] == 'c' else "认购"
                    nc = DynamicContractBuilder.get_atm(new_cn, spot, expiry, cur_dt)
                    np_ = Position(nc, nc["premium"], ts, "long")
                    positions.append(np_)
                    self._trade("反手开仓", np_, ts, date_str,
                                f"反手→{new_cn}", expiry_info=exp_label)
                    logger.info(f"{date_str} {ts} 【反手开仓】{nc['name']} "
                                f"K={nc['strike']} P={nc['premium']:.4f}")
                    reversed_ = True
                    hedge_confirm = 0

            # 对冲（连续2次确认）
            if not hedged:
                lp = next((p for p in positions if p.direction == "long"), None)
                if lp and lp.pnl_pct() > PROFIT_THRESH:
                    hedge_confirm += 1
                    if hedge_confirm >= HEDGE_CONFIRM_N:
                        hc = DynamicContractBuilder.get_otm(
                            lp.contract["type"], spot, expiry, cur_dt, 0.03)
                        if hc:
                            hp = Position(hc, hc["premium"], ts, "short")
                            positions.append(hp)
                            income = hc["premium"] * MULTIPLIER - COMMISSION
                            self._trade("卖出对冲", hp, ts, date_str,
                                        f"连续{hedge_confirm}次浮盈{lp.pnl_pct()*100:.1f}%→对冲",
                                        pnl=0, expiry_info=exp_label)
                            logger.info(f"{date_str} {ts} 【对冲】卖{hc['name']} "
                                        f"K={hc['strike']} P={hc['premium']:.4f} "
                                        f"收入={income:.2f}")
                            hedged = True
                else:
                    hedge_confirm = 0

        day_pnl = sum(t["pnl"] for t in self.all_trades
                      if t["date"] == date_str and t["pnl"] != 0)
        self.capital += day_pnl
        ret = day_pnl / capital_at_open if capital_at_open else 0.0

        result = {
            "date":           date_str,
            "etf_open":       open_p,
            "etf_close":      bar["close"],
            "etf_chg":        round((bar["close"] - prev_close) / prev_close * 100, 3),
            "expiry":         expiry.strftime("%Y-%m-%d"),
            "expiry_label":   exp_label,
            "signal":         signal or "无",
            "day_trades":     sum(1 for t in self.all_trades if t["date"] == date_str),
            "reversed":       reversed_,
            "hedged":         hedged,
            "stop_loss":      stop_loss_triggered,
            "daily_pnl":      round(day_pnl, 2),
            "daily_return":   round(ret, 6),
            "total_value":    round(self.capital, 2),
            "data_source":    data_source,
            "intraday_detail": intraday_detail,
        }
        self.daily_results.append(result)
        return result

    def run(self):
        logger.info(f"=== 真实数据回测 v4 (3/19~3/30) ===  资金={INIT_CAPITAL:,.0f}")
        for i in range(1, len(self.etf)):
            r = self.run_day(i)
            src_tag = "[真]" if r["data_source"] == "真实分钟" else "[模]"
            logger.info(
                f"{r['date']} {src_tag} [{r['expiry_label']}]  信号={r['signal']:<4} "
                f"反手={r['reversed']}  对冲={r['hedged']}  止损={r['stop_loss']}  "
                f"盈亏={r['daily_pnl']:>8.2f}  净值={r['total_value']:>10.2f}"
            )
        return pd.DataFrame(self.daily_results), pd.DataFrame(self.all_trades)


# ── 构建日线数据（从分钟数据聚合）─────────────────────────────
def build_daily_from_minute(dates_with_files: List[str]) -> pd.DataFrame:
    rows = []
    for ds in dates_with_files:
        fname = os.path.join(UPLOAD_DIR, f"159915_{ds.replace('-','')}_1min.csv")
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        rows.append({
            "date":     ds,
            "date_obj": date.fromisoformat(ds),
            "open":     df["open"].iloc[0],
            "high":     df["high"].max(),
            "low":      df["low"].min(),
            "close":    df["close"].iloc[-1],
            "volume":   df["volume"].sum(),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ── 主流程 ────────────────────────────────────────────────────
print("=" * 68)
print("  创业板ETF期权 v4策略 真实分钟数据回测 (3/19~3/30)")
print("=" * 68)

# 找所有上传的1分钟文件
import glob
files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "159915_*_1min.csv")))
date_strs = []
for f in files:
    base = os.path.basename(f)
    ds = base.split("_")[1]
    date_strs.append(f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}")
print(f"找到 {len(date_strs)} 个交易日数据: {date_strs[0]} ~ {date_strs[-1]}")

# 构建日线ETF数据
etf_df = build_daily_from_minute(date_strs)
print(f"日线数据: {len(etf_df)} 条")
print(etf_df[["date","open","high","low","close"]].to_string(index=False))

# 加载分钟数据
minute_data = {}
for ds in date_strs:
    fname = os.path.join(UPLOAD_DIR, f"159915_{ds.replace('-','')}_1min.csv")
    df = pd.read_csv(fname, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["time_str"] = pd.to_datetime(df["day"]).dt.strftime("%H:%M")
    df = df.set_index("time_str")
    minute_data[ds] = df

print(f"\n分钟数据已加载 {len(minute_data)} 天")
print()

# 运行回测
bt = RealDataBacktesterV4(etf_df, minute_data)
daily_df, trades_df = bt.run()

# ── 绩效报告 ──────────────────────────────────────────────────
print("\n" + "═"*68)
print("      真实数据回测结果（v4策略，3/19~3/30）")
print("═"*68)

total_ret = (bt.capital - INIT_CAPITAL) / INIT_CAPITAL
n_days    = len(daily_df)
ann_ret   = (1+total_ret)**(250/n_days)-1 if n_days else 0
rets      = daily_df["daily_return"]
sharpe    = (rets.mean()-RISK_FREE/250)/rets.std()*np.sqrt(250) if rets.std()>0 else 0
cum       = (1+rets).cumprod()
max_dd    = ((cum-cum.cummax())/cum.cummax()).min()
win_rate  = (daily_df["daily_pnl"]>0).mean()
avg_win   = daily_df[daily_df["daily_pnl"]>0]["daily_pnl"].mean() if (daily_df["daily_pnl"]>0).any() else 0
avg_loss  = abs(daily_df[daily_df["daily_pnl"]<0]["daily_pnl"].mean()) if (daily_df["daily_pnl"]<0).any() else 1
plr       = avg_win / avg_loss

n_open  = (trades_df["action"]=="买入开仓").sum() if len(trades_df) else 0
n_rev   = (trades_df["action"]=="反手开仓").sum() if len(trades_df) else 0
n_hedge = (trades_df["action"]=="卖出对冲").sum() if len(trades_df) else 0
n_sl    = (trades_df["action"]=="止损平仓").sum() if len(trades_df) else 0

print(f"\n初始资金: ¥{INIT_CAPITAL:>12,.2f}")
print(f"最终资金: ¥{bt.capital:>12,.2f}")
print(f"总收益率: {total_ret*100:>+.2f}%  (年化 {ann_ret*100:>+.1f}%)")
print(f"夏普比率: {sharpe:>+.4f}")
print(f"最大回撤: {max_dd*100:.2f}%")
print(f"胜率:    {win_rate*100:.1f}%   盈亏比: {plr:.3f}")
print(f"\n开仓次数: {n_open}  反手: {n_rev}  对冲: {n_hedge}  止损: {n_sl}")
print(f"交易日数: {n_days}")

print("\n─── 每日明细 ───────────────────────────────────────────")
print(f"{'日期':<12} {'数据':<6} {'信号':<5} {'日内变动':>8} {'日盈亏':>10} {'净值':>12} {'备注'}")
for _, r in daily_df.iterrows():
    src = "[真]" if r["data_source"] == "真实分钟" else "[模]"
    flags = []
    if r["reversed"]:   flags.append("反手")
    if r["hedged"]:     flags.append("对冲")
    if r["stop_loss"]:  flags.append("止损")
    note = " ".join(flags) if flags else "-"
    print(f"{r['date']:<12} {src:<6} {r['signal']:<5} "
          f"{r['etf_chg']:>+7.2f}% {r['daily_pnl']:>10.2f} "
          f"{r['total_value']:>12.2f} {note}")

# 保存结果
perf = {
    "initial_capital": INIT_CAPITAL,
    "final_capital":   round(bt.capital, 2),
    "total_return":    round(total_ret*100, 2),
    "annual_return":   round(ann_ret*100, 2),
    "sharpe":          round(sharpe, 4),
    "max_drawdown":    round(max_dd*100, 2),
    "win_rate":        round(win_rate*100, 1),
    "profit_loss_ratio": round(plr, 3),
    "n_trades":        len(trades_df),
    "n_open":          int(n_open),
    "n_reverse":       int(n_rev),
    "n_hedge":         int(n_hedge),
    "n_stop_loss":     int(n_sl),
    "n_days":          n_days,
    "period":          f"{date_strs[0]} ~ {date_strs[-1]}",
}

with open(os.path.join(OUTPUT_DIR, "backtest_realdata_v4_perf.json"), "w", encoding="utf-8") as f:
    json.dump(perf, f, ensure_ascii=False, indent=2)

daily_df.drop(columns=["intraday_detail"], errors="ignore").to_csv(
    os.path.join(OUTPUT_DIR, "backtest_realdata_v4_daily.csv"), index=False, encoding="utf-8-sig")
trades_df.to_csv(
    os.path.join(OUTPUT_DIR, "backtest_realdata_v4_trades.csv"), index=False, encoding="utf-8-sig")

print(f"\n结果已保存至 outputs/backtest_realdata_v4_*.csv/json")

# ── 生成HTML报告（内嵌式）─────────────────────────────────────
print("\n生成可视化报告...")

# 准备图表数据
dates_list = daily_df["date"].tolist()
pnl_list   = daily_df["daily_pnl"].tolist()
nav_list   = [(v/INIT_CAPITAL) for v in daily_df["total_value"].tolist()]
signals    = daily_df["signal"].tolist()

# 日内明细（所有交易日拼合）
trades_html = ""
if len(trades_df) > 0:
    for _, t in trades_df.iterrows():
        pnl_str = f"<span style='color:{'#22c55e' if t['pnl']>0 else '#ef4444' if t['pnl']<0 else '#94a3b8'}'>{t['pnl']:+.2f}</span>" if t['pnl'] != 0 else "-"
        trades_html += f"""
        <tr>
          <td>{t['date']}</td><td>{t['time']}</td>
          <td><span class="badge {'buy' if '买' in t['action'] else 'sell'}">{t['action']}</span></td>
          <td>{t['option_name']}</td>
          <td>{t['strike']}</td>
          <td>{t['price']:.4f}</td>
          <td>{pnl_str}</td>
          <td style='color:#94a3b8;font-size:11px'>{t['reason']}</td>
        </tr>"""

# 每日明细行
daily_html = ""
for _, r in daily_df.iterrows():
    src = "真实" if r["data_source"] == "真实分钟" else "模拟"
    flags = []
    if r["reversed"]:  flags.append("<span class='badge sell'>反手</span>")
    if r["hedged"]:    flags.append("<span class='badge'>对冲</span>")
    if r["stop_loss"]: flags.append("<span class='badge sell'>止损</span>")
    note = " ".join(flags) if flags else "-"
    chg_color = "#22c55e" if r["etf_chg"] > 0 else "#ef4444"
    pnl_color = "#22c55e" if r["daily_pnl"] > 0 else "#ef4444" if r["daily_pnl"] < 0 else "#94a3b8"
    daily_html += f"""
    <tr>
      <td>{r['date']}</td>
      <td><span style='color:#94a3b8;font-size:11px'>[{src}]</span></td>
      <td>{r['signal']}</td>
      <td style='color:{chg_color}'>{r['etf_chg']:+.2f}%</td>
      <td>{r['etf_open']:.3f}</td>
      <td>{r['etf_close']:.3f}</td>
      <td style='color:{pnl_color}'>{r['daily_pnl']:+.2f}</td>
      <td>{r['total_value']:,.2f}</td>
      <td>{note}</td>
      <td style='color:#94a3b8;font-size:11px'>{r['expiry_label']}</td>
    </tr>"""

# 构建最终HTML
html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>创业板ETF期权 v4策略 真实数据回测报告 (3/19~3/30)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#0f172a; color:#e2e8f0; font-size:13px; line-height:1.6 }}
  .container {{ max-width:1400px; margin:0 auto; padding:24px }}
  h1 {{ font-size:22px; font-weight:700; color:#f1f5f9; margin-bottom:4px }}
  .subtitle {{ color:#64748b; font-size:13px; margin-bottom:24px }}
  .grid {{ display:grid; gap:16px }}
  .grid-4 {{ grid-template-columns:repeat(4,1fr) }}
  .grid-2 {{ grid-template-columns:repeat(2,1fr) }}
  .card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:20px }}
  .card h2 {{ font-size:14px; color:#94a3b8; margin-bottom:12px; font-weight:500 }}
  .metric {{ text-align:center }}
  .metric .val {{ font-size:26px; font-weight:700; }}
  .metric .lbl {{ color:#64748b; font-size:11px; margin-top:4px }}
  .pos {{ color:#22c55e }} .neg {{ color:#ef4444 }} .neu {{ color:#94a3b8 }}
  .chart-wrap {{ position:relative; height:220px }}
  table {{ width:100%; border-collapse:collapse; font-size:12px }}
  th {{ color:#64748b; font-weight:500; padding:8px 6px; border-bottom:1px solid #334155; text-align:left }}
  td {{ padding:7px 6px; border-bottom:1px solid #1e293b }}
  tr:hover td {{ background:#334155 }}
  .badge {{ display:inline-block; padding:2px 6px; border-radius:4px; font-size:11px;
            background:#3b4a6b; color:#93c5fd }}
  .badge.buy {{ background:#14532d; color:#4ade80 }}
  .badge.sell {{ background:#450a0a; color:#f87171 }}
  .tag {{ display:inline-block; padding:1px 5px; border-radius:3px; font-size:10px;
          background:#1e3a5f; color:#60a5fa }}
  .info-bar {{ background:#1e293b; border:1px solid #334155; border-radius:8px;
               padding:12px 16px; margin-bottom:16px; font-size:12px; color:#94a3b8 }}
  .info-bar span {{ color:#e2e8f0; font-weight:500 }}
</style>
</head>
<body>
<div class="container">
  <h1>创业板ETF期权 v4策略 — 真实分钟数据回测</h1>
  <div class="subtitle">回测周期: {date_strs[0]} ~ {date_strs[-1]} &nbsp;|&nbsp; 数据来源: 真实1分钟K线 &nbsp;|&nbsp; 策略: 三阶段日内（四项优化）</div>

  <div class="info-bar">
    策略参数:
    <span>入场时间=09:45</span> &nbsp;|&nbsp;
    <span>开仓阈值=0.5%</span> &nbsp;|&nbsp;
    <span>反手阈值=0.8%</span> &nbsp;|&nbsp;
    <span>止损=35%</span> &nbsp;|&nbsp;
    <span>对冲浮盈=5%(连续2次)</span> &nbsp;|&nbsp;
    <span>强平=14:45</span>
  </div>

  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card metric">
      <div class="val {'pos' if total_ret>=0 else 'neg'}">{total_ret*100:+.2f}%</div>
      <div class="lbl">总收益率</div>
    </div>
    <div class="card metric">
      <div class="val {'pos' if sharpe>=0 else 'neg'}">{sharpe:+.2f}</div>
      <div class="lbl">夏普比率</div>
    </div>
    <div class="card metric">
      <div class="val neg">{max_dd*100:.2f}%</div>
      <div class="lbl">最大回撤</div>
    </div>
    <div class="card metric">
      <div class="val">{win_rate*100:.0f}%</div>
      <div class="lbl">胜率</div>
    </div>
  </div>

  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card metric">
      <div class="val neu">{n_days}</div>
      <div class="lbl">交易日</div>
    </div>
    <div class="card metric">
      <div class="val neu">{n_open}</div>
      <div class="lbl">开仓次数</div>
    </div>
    <div class="card metric">
      <div class="val neu">{n_rev}</div>
      <div class="lbl">反手次数</div>
    </div>
    <div class="card metric">
      <div class="val neu">{n_sl}</div>
      <div class="lbl">止损次数</div>
    </div>
  </div>

  <div class="grid grid-2" style="margin-bottom:16px">
    <div class="card">
      <h2>净值曲线</h2>
      <div class="chart-wrap"><canvas id="navChart"></canvas></div>
    </div>
    <div class="card">
      <h2>每日盈亏 (元)</h2>
      <div class="chart-wrap"><canvas id="pnlChart"></canvas></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <h2>每日交易明细</h2>
    <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>日期</th><th>数据</th><th>信号</th><th>ETF涨跌</th>
          <th>开盘</th><th>收盘</th><th>日盈亏</th><th>净值</th>
          <th>操作</th><th>合约</th>
        </tr>
      </thead>
      <tbody>{daily_html}</tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <h2>全部交易记录</h2>
    <div style="overflow-x:auto; max-height:400px; overflow-y:auto">
    <table>
      <thead>
        <tr>
          <th>日期</th><th>时间</th><th>操作</th><th>合约名称</th>
          <th>行权价</th><th>权利金</th><th>盈亏</th><th>原因</th>
        </tr>
      </thead>
      <tbody>{trades_html if trades_html else '<tr><td colspan="8" style="text-align:center;color:#64748b">无交易记录</td></tr>'}</tbody>
    </table>
    </div>
  </div>
</div>

<script>
const dates = {json.dumps(dates_list)};
const navs  = {json.dumps([round(n,6) for n in nav_list])};
const pnls  = {json.dumps([round(p,2) for p in pnl_list])};

const commonOpts = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{
    x:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:10}}}}}},
    y:{{grid:{{color:'#334155'}},ticks:{{color:'#64748b',font:{{size:10}}}}}}
  }}
}};

new Chart(document.getElementById('navChart'), {{
  type:'line',
  data:{{labels:dates, datasets:[{{
    data:navs, borderColor:'#3b82f6', backgroundColor:'rgba(59,130,246,0.08)',
    fill:true, tension:0.3, pointRadius:3, pointBackgroundColor:'#3b82f6',
    borderWidth:2, pointHoverRadius:5
  }}]}},
  options:{{...commonOpts,
    scales:{{...commonOpts.scales,
      y:{{...commonOpts.scales.y,
        ticks:{{...commonOpts.scales.y.ticks,
          callback:v=>(v*100).toFixed(2)+'%'}}}}}}
  }}
}});

new Chart(document.getElementById('pnlChart'), {{
  type:'bar',
  data:{{labels:dates, datasets:[{{
    data:pnls,
    backgroundColor: pnls.map(v=>v>0?'rgba(34,197,94,0.7)':'rgba(239,68,68,0.7)'),
    borderColor: pnls.map(v=>v>0?'#22c55e':'#ef4444'),
    borderWidth:1
  }}]}},
  options:commonOpts
}});
</script>
</body>
</html>"""

report_path = os.path.join(OUTPUT_DIR, "backtest_realdata_v4.html")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML报告已生成: {report_path}")
print("\n" + "═"*68)
print("回测完成！")
print("═"*68)
