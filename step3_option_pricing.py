# -*- coding: utf-8 -*-
"""
Step 3: 期权定价与风险指标计算
- 正向定价 (Black-76)         py_vollib.black.black
- 逆向推IV                    py_vollib.black.implied_volatility
- Greeks                      py_vollib.black.greeks.analytical
- SVI 波动率曲面校准
- 校验价格
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

from py_vollib.black import black
from py_vollib.black.implied_volatility import implied_volatility
from py_vollib.black.greeks.analytical import delta, gamma, vega, theta, rho
from scipy.optimize import minimize, differential_evolution

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 读取Step1产出的数据
# ================================================================
etf_df = pd.read_csv(os.path.join(OUTPUT_DIR, "etf_159915_daily_2026Q1.csv"), parse_dates=["date"])
option_df = pd.read_csv(os.path.join(OUTPUT_DIR, "option_159915_contracts.csv"))

# 取最新的收盘价作为标的价格（用于计算）
latest_etf_close = etf_df["close"].iloc[-1]
print(f"ETF最新收盘价(2026-03-30): {latest_etf_close}")

# ================================================================
# Black-76 模型 (期货期权 / 股权期权用前向价格 F=S 无分红简化)
# ================================================================
# 参数说明:
#   flag: 'c' = call / 'p' = put
#   F:    forward price (标的价格，ETF期权用spot price近似)
#   K:    strike price
#   t:    time to expiry (年)
#   r:    risk-free rate (年化)
#   sigma:volatility (年化)

def days_to_expiry_years(expiry_date_str: str, current_date_str: str = "2026-03-30") -> float:
    """计算距到期日的年化时间"""
    from datetime import datetime
    expiry = datetime.strptime(str(expiry_date_str)[:10], "%Y-%m-%d")
    current = datetime.strptime(current_date_str, "%Y-%m-%d")
    days = (expiry - current).days
    return max(days, 1) / 365.0

# 无风险利率（中国1年期国债收益率约1.8%）
RISK_FREE_RATE = 0.018

print("\n" + "=" * 60)
print("Step 3a: Black-76 正向定价")
print("=" * 60)

# 准备期权数据用于定价
option_pricing = option_df.copy()
option_pricing["最后交易日"] = pd.to_datetime(option_pricing["最后交易日"])
option_pricing["T"] = option_pricing["最后交易日"].apply(
    lambda x: days_to_expiry_years(x.strftime("%Y-%m-%d"))
)
option_pricing["F"] = latest_etf_close  # 用spot价格近似forward
option_pricing["r"] = RISK_FREE_RATE

# 假设隐含波动率约30%（初始估计，后面逆向推导）
INIT_SIGMA = 0.30

def calc_flag(contract_type: str) -> str:
    return 'c' if contract_type == '认购' else 'p'

# 计算Black-76理论价格
black76_prices = []
for _, row in option_pricing.iterrows():
    flag = calc_flag(row["合约类型"])
    F = row["F"]
    K = row["行权价"]
    t = row["T"]
    r = row["r"]
    try:
        price = black(flag, F, K, t, r, INIT_SIGMA)
        black76_prices.append(round(price, 6))
    except Exception as e:
        black76_prices.append(np.nan)

option_pricing["Black76_Price"] = black76_prices

print(f"\nBlack-76 定价完成（标的={latest_etf_close}, σ={INIT_SIGMA}, r={RISK_FREE_RATE}）")
print("\n样本合约定价（前20条）:")
cols = ["合约简称", "合约类型", "行权价", "T", "Black76_Price"]
print(option_pricing[cols].head(20).to_string(index=False))

# ================================================================
# Step 3b: 逆向推导隐含波动率 (Implied Volatility)
# 使用市场价格 = 前结算价作为市场价
# ================================================================
print("\n" + "=" * 60)
print("Step 3b: 逆向推导隐含波动率 (IV)")
print("=" * 60)

iv_list = []
for _, row in option_pricing.iterrows():
    flag = calc_flag(row["合约类型"])
    F = row["F"]
    K = row["行权价"]
    t = row["T"]
    r = row["r"]

    # 市场价：使用前结算价（若有）或用Black76价格模拟
    market_price = row.get("前结算价", row["Black76_Price"])
    if pd.isna(market_price) or market_price <= 0:
        market_price = row["Black76_Price"]

    try:
        if pd.isna(market_price) or market_price <= 0.0001:
            iv_list.append(np.nan)
        else:
            iv = implied_volatility(market_price, F, K, t, r, flag)
            iv_list.append(round(iv, 6))
    except Exception:
        iv_list.append(np.nan)

option_pricing["IV"] = iv_list

valid_iv = option_pricing["IV"].dropna()
print(f"\nIV计算完成: {len(valid_iv)}/{len(option_pricing)} 个合约有效")
print(f"IV范围: {valid_iv.min():.4f} - {valid_iv.max():.4f}")
print(f"IV均值: {valid_iv.mean():.4f}")

# ================================================================
# Step 3c: 计算Greeks
# ================================================================
print("\n" + "=" * 60)
print("Step 3c: 计算 Greeks (Delta/Gamma/Vega/Theta/Rho)")
print("=" * 60)

greek_rows = []
for _, row in option_pricing.iterrows():
    flag = calc_flag(row["合约类型"])
    F = row["F"]
    K = row["行权价"]
    t = row["T"]
    r = row["r"]
    sigma = row["IV"] if not pd.isna(row["IV"]) else INIT_SIGMA

    try:
        d = delta(flag, F, K, t, r, sigma)
        g = gamma(flag, F, K, t, r, sigma)
        v = vega(flag, F, K, t, r, sigma)
        th = theta(flag, F, K, t, r, sigma)
        rh = rho(flag, F, K, t, r, sigma)
    except Exception:
        d = g = v = th = rh = np.nan

    greek_rows.append({
        "合约代码": row["合约代码"],
        "合约简称": row["合约简称"],
        "合约类型": row["合约类型"],
        "行权价": K,
        "T": round(t, 4),
        "IV": sigma,
        "Black76_Price": row["Black76_Price"],
        "Delta": round(d, 6) if not np.isnan(d) else np.nan,
        "Gamma": round(g, 6) if not np.isnan(g) else np.nan,
        "Vega": round(v, 6) if not np.isnan(v) else np.nan,
        "Theta": round(th, 6) if not np.isnan(th) else np.nan,
        "Rho": round(rh, 6) if not np.isnan(rh) else np.nan,
    })

greeks_df = pd.DataFrame(greek_rows)
print("\n样本Greeks（当月合约，前20条）:")
print(greeks_df.head(20).to_string(index=False))

# ================================================================
# Step 3d: SVI 波动率曲面校准
# ================================================================
print("\n" + "=" * 60)
print("Step 3d: SVI 波动率曲面校准 (Stochastic Volatility Inspired)")
print("=" * 60)

def svi_raw(k, a, b, rho_svi, m, sigma_svi):
    """
    SVI Raw参数化
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    其中 k = log(K/F)，w = IV^2 * T (total implied variance)
    """
    return a + b * (rho_svi * (k - m) + np.sqrt((k - m)**2 + sigma_svi**2))

def svi_to_iv(k, t, a, b, rho_svi, m, sigma_svi):
    """从SVI参数计算IV"""
    w = svi_raw(k, a, b, rho_svi, m, sigma_svi)
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / t)

def calibrate_svi(strikes, ivs, F, T):
    """
    SVI 曲面校准
    Returns: (a, b, rho, m, sigma) SVI参数
    """
    # log-moneyness
    K_arr = np.array(strikes, dtype=float)
    iv_arr = np.array(ivs, dtype=float)
    k_arr = np.log(K_arr / F)

    # 总方差
    w_arr = iv_arr**2 * T

    def objective(params):
        a, b, rho_p, m, sigma_p = params
        w_fit = svi_raw(k_arr, a, b, rho_p, m, sigma_p)
        return np.sum((w_fit - w_arr)**2)

    def butterfly_constraint(params):
        """确保无蝶式套利：SVI曲线凸性约束"""
        a, b, rho_p, m, sigma_p = params
        return b * (1 - abs(rho_p))  # > 0

    bounds = [
        (-0.1, 0.5),   # a
        (0.001, 1.0),  # b
        (-0.999, 0.999), # rho
        (-0.5, 0.5),   # m
        (0.001, 1.0),  # sigma
    ]

    constraints = [{"type": "ineq", "fun": butterfly_constraint}]

    # 初始参数
    w_mean = np.mean(w_arr)
    x0 = [w_mean * 0.8, 0.1, -0.3, 0.0, 0.1]

    try:
        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10}
        )
        if result.success:
            return result.x, result.fun
        else:
            # 尝试全局优化
            result2 = differential_evolution(
                objective, bounds,
                maxiter=500, tol=1e-8, seed=42
            )
            return result2.x, result2.fun
    except Exception as e:
        return x0, np.inf

# 按到期月份分组做SVI校准
svi_results = {}
greeks_df_ext = greeks_df.copy()
greeks_df_ext["expiry"] = option_pricing["最后交易日"].values
greeks_df_ext["T"] = option_pricing["T"].values

print("\n按到期月份进行SVI曲面校准:")

for expiry, grp in greeks_df_ext.groupby("expiry"):
    # 过滤有效IV的合约
    valid = grp[grp["IV"].notna() & (grp["IV"] > 0.01) & (grp["IV"] < 5.0)]
    if len(valid) < 4:
        print(f"  {expiry}: 有效数据不足({len(valid)}个)，跳过")
        continue

    T_val = valid["T"].iloc[0]
    F_val = latest_etf_close

    strikes = valid["行权价"].values
    ivs = valid["IV"].values

    params, residual = calibrate_svi(strikes, ivs, F_val, T_val)
    a, b, rho_p, m, sigma_p = params

    # 生成拟合曲线
    k_range = np.linspace(np.log(strikes.min()/F_val) - 0.1,
                           np.log(strikes.max()/F_val) + 0.1, 50)
    svi_ivs = svi_to_iv(k_range, T_val, a, b, rho_p, m, sigma_p)

    svi_results[str(expiry)[:10]] = {
        "params": {"a": round(a,6), "b": round(b,6), "rho": round(rho_p,6),
                   "m": round(m,6), "sigma": round(sigma_p,6)},
        "T": T_val,
        "residual_mse": round(residual / max(len(valid),1), 8),
        "n_contracts": len(valid),
        "strike_range": [float(strikes.min()), float(strikes.max())]
    }

    print(f"\n  到期日: {str(expiry)[:10]} | T={T_val:.3f}年 | 样本数={len(valid)}")
    print(f"  SVI参数: a={a:.4f}, b={b:.4f}, ρ={rho_p:.4f}, m={m:.4f}, σ={sigma_p:.4f}")
    print(f"  MSE残差: {residual/max(len(valid),1):.2e}")

    # 计算SVI拟合IV（ATM附近）
    atm_k = 0.0
    atm_svi_iv = svi_to_iv(atm_k, T_val, a, b, rho_p, m, sigma_p)
    print(f"  ATM SVI-IV: {atm_svi_iv:.4f} ({atm_svi_iv*100:.2f}%)")

# ================================================================
# Step 3e: 校验价格 (Check for arbitrage violations)
# ================================================================
print("\n" + "=" * 60)
print("Step 3e: 价格校验（无套利检验）")
print("=" * 60)

def check_arbitrage(df, F, r, expiry_col="expiry"):
    violations = []
    for expiry, grp in df.groupby(expiry_col):
        T = grp["T"].iloc[0]
        calls = grp[grp["合约类型"] == "认购"].sort_values("行权价")
        puts = grp[grp["合约类型"] == "认沽"].sort_values("行权价")

        # 检查 put-call parity (简化版)
        for _, call_row in calls.iterrows():
            K = call_row["行权价"]
            # 找对应put
            matching_put = puts[puts["行权价"] == K]
            if matching_put.empty:
                continue
            put_row = matching_put.iloc[0]

            C = call_row["Black76_Price"]
            P = put_row["Black76_Price"]

            if pd.isna(C) or pd.isna(P):
                continue

            # Put-Call Parity: C - P = (F - K) * e^(-rT)
            pcp_lhs = C - P
            pcp_rhs = (F - K) * np.exp(-r * T)
            pcp_error = abs(pcp_lhs - pcp_rhs)

            if pcp_error > 0.005:  # 允许0.5%误差
                violations.append({
                    "expiry": str(expiry)[:10],
                    "strike": K,
                    "call_price": C,
                    "put_price": P,
                    "parity_error": round(pcp_error, 6)
                })

        # 检查认购期权单调性（行权价升高，看涨价格应降低）
        if len(calls) > 1:
            call_prices = calls["Black76_Price"].values
            for i in range(len(call_prices) - 1):
                if call_prices[i+1] > call_prices[i] + 0.001:
                    violations.append({
                        "expiry": str(expiry)[:10],
                        "type": "call_monotone_violation",
                        "detail": f"K={calls.iloc[i]['行权价']}→{calls.iloc[i+1]['行权价']}"
                    })

    return violations

check_df = greeks_df_ext.copy()
check_df["Black76_Price"] = greeks_df["Black76_Price"].values

violations = check_arbitrage(check_df, latest_etf_close, RISK_FREE_RATE)
if violations:
    print(f"发现 {len(violations)} 个潜在套利违规:")
    for v in violations[:10]:
        print(f"  {v}")
else:
    print("✓ 价格校验通过：未发现明显套利机会")

# ================================================================
# 保存所有计算结果
# ================================================================
# 完整Greeks结果
greeks_output = greeks_df_ext.copy()
greeks_output.to_csv(os.path.join(OUTPUT_DIR, "option_159915_greeks.csv"),
                      index=False, encoding="utf-8-sig")

# SVI参数
import json
with open(os.path.join(OUTPUT_DIR, "svi_parameters.json"), "w", encoding="utf-8") as f:
    json.dump(svi_results, f, ensure_ascii=False, indent=2)

print(f"\n✓ Greeks数据已保存: outputs/option_159915_greeks.csv")
print(f"✓ SVI参数已保存: outputs/svi_parameters.json")
print("\n✓ Step 3 完成")
