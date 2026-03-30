# -*- coding: utf-8 -*-
"""
Step 5 v3.1: ETF期权三阶段策略 — 真实合约换月模拟（改进版）
核心特性：
  - 动态选择当月/次月合约（距到期≤5交易日换月）
  - 行权价0.05元档位，精确到0.001元（真实市场精度）
  - 反手截止时间 ≤13:30，避免末尾换手陷阱
  - IV 期限结构调整（近月溢价）
  - 对冲双重计算Bug已修复
"""

import sys, os, json, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Tuple
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
        logging.FileHandler(os.path.join(OUTPUT_DIR, "backtest_v3.log"),
                            encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bt_v3")
np.random.seed(42)

# ── 全局参数 ──────────────────────────────────────────────────
RISK_FREE    = 0.018
MULTIPLIER   = 10000
COMMISSION   = 5.0
INIT_CAPITAL = 200_000.0
OPEN_THRESH  = 0.005    # 开盘>0.5% 触发
REV_THRESH   = 0.005    # 偏离>0.5% 反手
PROFIT_THRESH= 0.05     # 盈利>5% 对冲
ROLL_DAYS    = 5        # 距到期≤5交易日换月

INTRADAY_TIMES = ["09:30","10:00","10:30","11:00","11:30",
                  "13:00","13:30","14:00","14:30","14:45"]
CLEAR_TIME = "14:45"

MONTH_CN = {1:"1月",2:"2月",3:"3月",4:"4月",5:"5月",
            6:"6月",7:"7月",8:"8月",9:"9月",10:"10月",
            11:"11月",12:"12月"}

# ── 2026年每月期权到期日（第四个周三）─────────────────────────
# ETF期权到期日：到期月第四个周三（如遇节假日提前）
EXPIRY_SCHEDULE = {
    (2026,  1): date(2026,  1, 28),   # 第4个周三
    (2026,  2): date(2026,  2, 25),
    (2026,  3): date(2026,  3, 25),
    (2026,  4): date(2026,  4, 22),
    (2026,  5): date(2026,  5, 27),
    (2026,  6): date(2026,  6, 24),
    (2026,  9): date(2026,  9, 23),
    (2026, 12): date(2026, 12, 23),
}

def get_expiry(year: int, month: int) -> date:
    return EXPIRY_SCHEDULE.get((year, month),
           date(year, month, 28))   # fallback

def trading_days_to_expiry(trade_date: date, expiry: date,
                            trading_set: set) -> int:
    d, cnt = trade_date, 0
    while d < expiry:
        d += timedelta(days=1)
        if d in trading_set:
            cnt += 1
    return cnt

def choose_expiry(trade_date: date, trading_set: set) -> date:
    """选当月到期合约；如距到期≤ROLL_DAYS 则用次月"""
    curr_exp = get_expiry(trade_date.year, trade_date.month)
    td = trading_days_to_expiry(trade_date, curr_exp, trading_set)
    if td <= ROLL_DAYS:
        nm = trade_date.month + 1 if trade_date.month < 12 else 1
        ny = trade_date.year if trade_date.month < 12 else trade_date.year + 1
        return get_expiry(ny, nm)
    return curr_exp

# ── IV 期限结构估计（基于SVI校准结果）──────────────────────────
# 我们只有4月后的SVI参数，对短期合约用外推
SVI_BASE_IV = {"call": 0.52, "put": 0.46}   # 4月ATM IV（来自greeks数据）

def estimate_iv(flag: str, K: float, F: float, T: float) -> float:
    """
    用简化SVI给出IV估计：
      base_iv (根据方向) + 期限调整 + 偏度调整
    """
    base = SVI_BASE_IV["call"] if flag == 'c' else SVI_BASE_IV["put"]
    # 期限结构：短期溢价（近月波动率更高）
    # 参考：4月(T≈0.063)→0.52, 9月(T≈0.485)→更低
    # 用简单幂律: iv(T) = base * (0.063/T)^0.08
    T_ref = 0.063
    term_adj = (T_ref / max(T, 0.01)) ** 0.08
    # 偏度：虚值认沽溢价（skew）
    moneyness = np.log(F / K)   # + = OTM put/ITM call
    skew = -0.15 * moneyness
    iv = base * term_adj + skew
    return max(0.15, min(iv, 1.20))

# ── Black-76 定价 ─────────────────────────────────────────────
def bk76(flag: str, F: float, K: float, T: float,
         r: float = RISK_FREE, sigma: float = 0.35) -> float:
    try:
        p = black(flag, F, K, max(T, 1/365), r, max(sigma, 0.05))
        return max(p, 0.0001)
    except Exception:
        iv = max(0, F-K) if flag=='c' else max(0, K-F)
        return max(iv + F*sigma*np.sqrt(max(T,1/365))*0.15, 0.0001)

def T_years(expiry: date, current_dt: datetime) -> float:
    exp_dt = datetime(expiry.year, expiry.month, expiry.day, 15, 0)
    delta  = exp_dt - current_dt
    return max(delta.total_seconds() / (365 * 86400), 0.5/365)

# ── 动态合约生成器 ─────────────────────────────────────────────
class DynamicContractBuilder:
    """
    根据当日ETF价格 + 到期日，动态构造合约字典。
    行权价以0.05元为间距（真实市场档位），精确到0.001元，生成ATM周围±4档。
    """
    @staticmethod
    def build(cn_type: str, spot: float, expiry: date) -> List[dict]:
        flag   = 'c' if cn_type == "认购" else 'p'
        month  = MONTH_CN[expiry.month]
        # 行权价精确到0.001元（真实精度），以0.05为档位搜索ATM，再生成±4档网格
        atm_k  = round(round(spot / 0.05) * 0.05, 3)
        strikes = [round(atm_k + i*0.05, 3) for i in range(-4, 5)]
        contracts = []
        for K in strikes:
            if K <= 0:
                continue
            code = f"159915{cn_type[0]}{expiry.strftime('%y%m')}{int(K*1000):05d}"
            name = f"创业板ETF{cn_type[0]}{month}{int(K*1000)}"
            contracts.append({
                "code": code, "name": name,
                "strike": K,  "expiry": expiry.strftime("%Y-%m-%d"),
                "type": cn_type, "flag": flag
            })
        return contracts

    @staticmethod
    def get_atm(cn_type: str, spot: float, expiry: date,
                current_dt: datetime) -> dict:
        contracts = DynamicContractBuilder.build(cn_type, spot, expiry)
        best = min(contracts, key=lambda c: abs(c["strike"] - spot))
        T = T_years(expiry, current_dt)
        iv = estimate_iv(best["flag"], best["strike"], spot, T)
        best["iv"] = iv
        best["premium"] = bk76(best["flag"], spot, best["strike"], T,
                               RISK_FREE, iv)
        return best

    @staticmethod
    def get_otm(cn_type: str, spot: float, expiry: date,
                current_dt: datetime, otm_pct: float = 0.03) -> Optional[dict]:
        contracts = DynamicContractBuilder.build(cn_type, spot, expiry)
        if cn_type == "认购":
            target = spot * (1 + otm_pct)
            cands  = [c for c in contracts if c["strike"] > spot]
        else:
            target = spot * (1 - otm_pct)
            cands  = [c for c in contracts if c["strike"] < spot]
        if not cands:
            cands = contracts
        best = min(cands, key=lambda c: abs(c["strike"] - target))
        if best["strike"] == DynamicContractBuilder.get_atm(
                cn_type, spot, expiry, current_dt)["strike"]:
            return None
        T = T_years(expiry, current_dt)
        iv = estimate_iv(best["flag"], best["strike"], spot, T)
        best["iv"] = iv
        best["premium"] = bk76(best["flag"], spot, best["strike"], T,
                               RISK_FREE, iv)
        return best

# ── 持仓对象 ─────────────────────────────────────────────────
class Position:
    def __init__(self, contract: dict, open_premium: float,
                 open_time: str, direction: str = "long"):
        self.contract      = contract
        self.open_premium  = open_premium
        self.current_price = open_premium
        self.open_time     = open_time
        self.direction     = direction
        self.qty           = 1

    def pnl(self) -> float:
        sign = 1 if self.direction == "long" else -1
        return sign * (self.current_price - self.open_premium) * MULTIPLIER

    def pnl_pct(self) -> float:
        return (self.current_price - self.open_premium) / self.open_premium

    def reprice(self, spot: float, current_dt: datetime):
        exp  = datetime.strptime(self.contract["expiry"], "%Y-%m-%d").date()
        T    = T_years(exp, current_dt)
        iv   = estimate_iv(self.contract["flag"],
                           self.contract["strike"], spot, T)
        self.current_price = bk76(self.contract["flag"], spot,
                                  self.contract["strike"], T,
                                  RISK_FREE, iv)
        self.contract["iv"] = iv

# ── 日内价格路径 ───────────────────────────────────────────────
def gen_intraday(bar: dict, seed_offset: int = 0) -> List[Tuple[str, float]]:
    rng = np.random.RandomState(42 + seed_offset)
    O, H, L, C = bar["open"], bar["high"], bar["low"], bar["close"]
    n = len(INTRADAY_TIMES)
    base  = np.linspace(O, C, n)
    noise = rng.normal(0, (H-L)*0.22, n)
    noise[0] = noise[-1] = 0
    prices = np.clip(base + noise, L, H)
    prices[0] = O; prices[-1] = C
    return list(zip(INTRADAY_TIMES, prices))

# ── 主回测引擎 ────────────────────────────────────────────────
class IntradayBacktester:
    def __init__(self, etf_df: pd.DataFrame):
        self.etf     = etf_df.reset_index(drop=True)
        self.trading_set = set(etf_df["date_obj"])
        self.capital = INIT_CAPITAL
        self.daily_results: List[dict] = []
        self.all_trades:    List[dict] = []

    def _trade(self, action: str, pos: Position, ts: str,
               date_str: str, reason: str, pnl: float = 0.0,
               expiry_info: str = ""):
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

        # ── 选择到期合约 ──────────────────────────────────────
        expiry = choose_expiry(trade_dt, self.trading_set)
        td_left = trading_days_to_expiry(trade_dt, expiry, self.trading_set)
        exp_label = f"{MONTH_CN[expiry.month]}合约(剩{td_left}交易日)"

        intraday = gen_intraday(
            {"open": open_p, "high": bar["high"],
             "low": bar["low"], "close": bar["close"]},
            seed_offset=idx
        )

        positions: List[Position] = []
        hedged = reversed_ = False
        capital_at_open = self.capital

        # ── 阶段1: 9:30 开盘探索 ─────────────────────────────
        change_rate = (open_p - prev_close) / prev_close
        signal = None
        if change_rate > OPEN_THRESH:
            signal = "认购"
        elif change_rate < -OPEN_THRESH:
            signal = "认沽"

        if signal:
            open_dt = datetime.strptime(f"{date_str} 09:30", "%Y-%m-%d %H:%M")
            c = DynamicContractBuilder.get_atm(signal, open_p, expiry, open_dt)
            pos = Position(c, c["premium"], "09:30", "long")
            positions.append(pos)
            self._trade("买入开仓", pos, "09:30", date_str,
                        f"开盘{'↑' if signal=='认购' else '↓'}{change_rate*100:.2f}%",
                        expiry_info=exp_label)
            logger.info(f"{date_str} 09:30 【开仓】{c['name']} K={c['strike']} "
                        f"P={c['premium']:.4f} T={T_years(expiry,open_dt)*365:.0f}天 "
                        f"到期={expiry} σ={c['iv']:.2f}")

        # ── 阶段2 & 3: 日内循环 ───────────────────────────────
        for ts, spot in intraday[1:]:
            cur_dt = datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M")

            for p in positions:
                p.reprice(spot, cur_dt)

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

            if not positions:
                continue

            # 阶段2: 反手验证（截止13:30，之后不再反手，避免末尾换手亏损）
            if not reversed_ and ts <= "13:30":
                lp = next((p for p in positions if p.direction=="long"), None)
                if lp:
                    chg = (spot - open_p) / open_p
                    rev_needed = (lp.contract["flag"]=='c' and chg < -REV_THRESH) or \
                                 (lp.contract["flag"]=='p' and chg >  REV_THRESH)
                    if rev_needed:
                        pnl_c = lp.pnl() - COMMISSION
                        self._trade("反手平仓", lp, ts, date_str,
                                    f"偏离{chg*100:.2f}%反手", pnl_c,
                                    expiry_info=exp_label)
                        logger.info(f"{date_str} {ts} 【反手平仓】{lp.contract['name']} "
                                    f"盈亏={pnl_c:.2f}")
                        positions.remove(lp)
                        new_cn = "认沽" if lp.contract["flag"]=='c' else "认购"
                        nc = DynamicContractBuilder.get_atm(new_cn, spot, expiry, cur_dt)
                        np_ = Position(nc, nc["premium"], ts, "long")
                        positions.append(np_)
                        self._trade("反手开仓", np_, ts, date_str,
                                    f"反手→{new_cn}", expiry_info=exp_label)
                        logger.info(f"{date_str} {ts} 【反手开仓】{nc['name']} "
                                    f"K={nc['strike']} P={nc['premium']:.4f}")
                        reversed_ = True

            # 阶段3: 对冲锁利
            if not hedged:
                lp = next((p for p in positions if p.direction=="long"), None)
                if lp and lp.pnl_pct() > PROFIT_THRESH:
                    hc = DynamicContractBuilder.get_otm(
                        lp.contract["type"], spot, expiry, cur_dt, 0.03)
                    if hc:
                        hp = Position(hc, hc["premium"], ts, "short")
                        positions.append(hp)
                        income = hc["premium"] * MULTIPLIER - COMMISSION
                        self._trade("卖出对冲", hp, ts, date_str,
                                    f"浮盈{lp.pnl_pct()*100:.1f}%→对冲",
                                    pnl=0, expiry_info=exp_label)  # 收入在平仓时结算，避免双重计算
                        logger.info(f"{date_str} {ts} 【对冲】卖{hc['name']} "
                                    f"K={hc['strike']} P={hc['premium']:.4f} "
                                    f"收入={income:.2f}")
                        hedged = True

        day_pnl = sum(t["pnl"] for t in self.all_trades
                      if t["date"] == date_str and t["pnl"] != 0)
        self.capital += day_pnl
        ret = day_pnl / capital_at_open if capital_at_open else 0.0

        result = {
            "date":         date_str,
            "etf_open":     open_p,
            "etf_close":    bar["close"],
            "expiry":       expiry.strftime("%Y-%m-%d"),
            "expiry_label": exp_label,
            "signal":       signal or "无",
            "day_trades":   sum(1 for t in self.all_trades if t["date"]==date_str),
            "reversed":     reversed_,
            "hedged":       hedged,
            "daily_pnl":    round(day_pnl, 2),
            "daily_return": round(ret, 6),
            "total_value":  round(self.capital, 2),
        }
        self.daily_results.append(result)
        return result

    def run(self):
        logger.info(f"=== 创业板ETF期权 日内三阶段 v3 ===  资金={INIT_CAPITAL:,.0f}")
        for i in range(1, len(self.etf)):
            r = self.run_day(i)
            exp_info = r["expiry_label"]
            logger.info(f"{r['date']}  [{exp_info}]  信号={r['signal']:<4} "
                        f"反手={r['reversed']}  对冲={r['hedged']}  "
                        f"盈亏={r['daily_pnl']:>8.2f}  净值={r['total_value']:>10.2f}")
        return pd.DataFrame(self.daily_results), pd.DataFrame(self.all_trades)


# ── 加载数据 ──────────────────────────────────────────────────
print("=" * 64)
print("加载 VeighNA 数据库...")

from vnpy.trader.database import get_database
from vnpy.trader.constant import Exchange, Interval

db   = get_database()
bars = db.load_bar_data("159915", Exchange.SZSE, Interval.DAILY,
                         datetime(2026,1,1), datetime(2026,3,31))

etf = pd.DataFrame([{
    "date":     b.datetime.strftime("%Y-%m-%d"),
    "date_obj": b.datetime.date(),
    "open":  b.open_price, "high": b.high_price,
    "low":   b.low_price,  "close": b.close_price,
}for b in bars]).sort_values("date").reset_index(drop=True)
print(f"ETF数据: {len(etf)} 条 ({etf['date'].iloc[0]} ~ {etf['date'].iloc[-1]})")

# 打印换月日程
print("\n【合约换月日程预览】")
trading_set = set(etf["date_obj"])
for _, row in etf.iterrows():
    d = row["date_obj"]
    exp = choose_expiry(d, trading_set)
    td  = trading_days_to_expiry(d, exp, trading_set)
    prev_exp = (get_expiry(d.year, d.month)
                if exp.month != d.month or exp.year != d.year else None)
    roll_flag = " ← 换月" if prev_exp and exp.month != d.month else ""
    if roll_flag or td <= 6:
        print(f"  {d}  使用{MONTH_CN[exp.month]}合约(到期={exp}, 剩{td}交易日){roll_flag}")

print()
bt = IntradayBacktester(etf)
daily_df, trades_df = bt.run()

# ── 绩效报告 ──────────────────────────────────────────────────
print("\n" + "═"*64)
print("      创业板ETF期权 三阶段策略 v3（动态换月）回测报告")
print("═"*64)

total_ret  = (bt.capital - INIT_CAPITAL) / INIT_CAPITAL
n_days     = len(daily_df)
ann_ret    = (1+total_ret)**(250/n_days)-1 if n_days else 0
rets       = daily_df["daily_return"]
sharpe     = (rets.mean()-RISK_FREE/250)/rets.std()*np.sqrt(250) if rets.std()>0 else 0
cum        = (1+rets).cumprod()
max_dd     = ((cum-cum.cummax())/cum.cummax()).min()
win_rate   = (daily_df["daily_pnl"]>0).mean()
avg_win    = daily_df[daily_df["daily_pnl"]>0]["daily_pnl"].mean() if (daily_df["daily_pnl"]>0).any() else 0
avg_loss   = abs(daily_df[daily_df["daily_pnl"]<0]["daily_pnl"].mean()) if (daily_df["daily_pnl"]<0).any() else 1
plr        = avg_win/avg_loss

n_open  = (trades_df["action"]=="买入开仓").sum()
n_rev   = (trades_df["action"]=="反手开仓").sum()
n_hedge = (trades_df["action"]=="卖出对冲").sum()
n_clear = trades_df["action"].isin(["卖出平仓","买入平仓"]).sum()

print(f"\n【基本信息】")
print(f"  初始资金: ¥{INIT_CAPITAL:>12,.2f}")
print(f"  期末净值: ¥{bt.capital:>12,.2f}")
print(f"  总收益率:   {total_ret*100:>+8.2f}%   年化: {ann_ret*100:>+.2f}%")

print(f"\n【风险收益】")
print(f"  夏普比率: {sharpe:>8.4f}")
print(f"  最大回撤: {max_dd*100:>8.2f}%")
print(f"  日均收益: {rets.mean()*100:>+8.4f}%")
print(f"  日收益波动: {rets.std()*100:>8.4f}%")

print(f"\n【交易统计】")
print(f"  总笔数: {len(trades_df):>5}  开仓:{n_open}  反手:{n_rev}  对冲:{n_hedge}  清仓:{n_clear}")
print(f"  胜率: {win_rate*100:.1f}%   盈亏比: {plr:.3f}")
print(f"  盈利天数:{(daily_df['daily_pnl']>0).sum()}  亏损:{(daily_df['daily_pnl']<0).sum()}  空仓:{(daily_df['daily_pnl']==0).sum()}")

print(f"\n【月度分析】")
daily_df["month"] = pd.to_datetime(daily_df["date"]).dt.to_period("M").astype(str)
monthly = daily_df.groupby("month").agg(
    天数=("date","count"),
    月盈亏=("daily_pnl","sum"),
    胜率=("daily_pnl",lambda x:(x>0).mean()),
    最大盈利=("daily_pnl","max"),
    最大亏损=("daily_pnl","min"),
    反手次数=("reversed","sum"),
    对冲次数=("hedged","sum"),
    月末净值=("total_value","last"),
).reset_index()
print(monthly.to_string(index=False))

print(f"\n【每日明细（含合约选择）】")
pd.set_option("display.max_columns",None)
pd.set_option("display.width",200)
print(daily_df[["date","etf_open","etf_close","expiry_label","signal",
               "day_trades","reversed","hedged","daily_pnl","total_value"]].to_string(index=False))

# 交易记录
print(f"\n【完整交易记录】")
print(trades_df[["date","time","action","option_name","direction","strike",
                  "expiry","price","premium_value","pnl","reason"]].to_string(index=False))

# 保存
perf = {
    "initial_capital": INIT_CAPITAL,
    "final_capital":   round(bt.capital,2),
    "total_return":    round(total_ret*100,4),
    "annual_return":   round(ann_ret*100,4),
    "sharpe":          round(sharpe,4),
    "max_drawdown":    round(max_dd*100,4),
    "win_rate":        round(win_rate*100,2),
    "profit_loss_ratio": round(plr,4),
    "n_trades": len(trades_df),
    "n_open": int(n_open), "n_reverse": int(n_rev),
    "n_hedge": int(n_hedge),
}
daily_df.to_csv(f"{OUTPUT_DIR}/backtest_v3_daily.csv",   index=False, encoding="utf-8-sig")
trades_df.to_csv(f"{OUTPUT_DIR}/backtest_v3_trades.csv", index=False, encoding="utf-8-sig")
monthly.to_csv(f"{OUTPUT_DIR}/backtest_v3_monthly.csv",  index=False, encoding="utf-8-sig")
with open(f"{OUTPUT_DIR}/backtest_v3_perf.json","w") as f:
    json.dump(perf, f, indent=2)

print(f"\n✓ 已保存: backtest_v3_{{daily,trades,monthly,perf}}")
