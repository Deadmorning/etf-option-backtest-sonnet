# -*- coding: utf-8 -*-
"""生成 v3 动态换月回测HTML报告"""
import pandas as pd, json, os, numpy as np

D = "outputs"
daily   = pd.read_csv(f"{D}/backtest_v3_daily.csv")
trades  = pd.read_csv(f"{D}/backtest_v3_trades.csv")
monthly = pd.read_csv(f"{D}/backtest_v3_monthly.csv")
etf     = pd.read_csv(f"{D}/etf_159915_daily_2026Q1.csv")
with open(f"{D}/backtest_v3_perf.json") as f: perf = json.load(f)

# 净值 & 回撤
rets = daily["daily_return"]
daily["nav"] = (1 + rets).cumprod()
cummax = daily["nav"].cummax()
daily["dd"] = (daily["nav"] - cummax) / cummax * 100

# 识别换月日
daily["roll"] = daily["expiry_label"].str.extract(r"(\d月)合约")[0]
roll_days = []
prev = None
for _, r in daily.iterrows():
    m = str(r["roll"])
    if prev and m != prev:
        roll_days.append(r["date"])
    prev = m

dates_js  = list(daily["date"])
nav_js    = [round(x,5) for x in daily["nav"]]
pnl_js    = [round(x,2) for x in daily["daily_pnl"]]
dd_js     = [round(x,5) for x in daily["dd"]]
etf_d_js  = list(etf["date"])
etf_c_js  = list(map(float, etf["close"]))

# 反手 & 对冲时间分布
rev = trades[trades["action"]=="反手平仓"]["time"].value_counts().sort_index()
hdg = trades[trades["action"]=="卖出对冲"]["time"].value_counts().sort_index()

# 月度 bar
m_labels = list(monthly["month"])
m_pnl    = [round(x,2) for x in monthly["月盈亏"]]
m_win    = [round(float(x)*100,1) for x in monthly["胜率"]]

# 合约分布（按expiry_label按月）
contract_use = trades[trades["action"]=="买入开仓"].groupby(
    trades["option_name"].str.extract(r"(认购|认沽)([1-9]?\d月)")[1].rename("月份")
).size().sort_index()

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>创业板ETF期权 动态换月策略 回测报告 v3</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI','PingFang SC',sans-serif;background:#0c1220;color:#e2e8f0}}
.hdr{{background:linear-gradient(135deg,#0a1f38,#0c1220);padding:28px 48px;border-bottom:1px solid #1e3a5f}}
.hdr h1{{font-size:22px;color:#60a5fa;margin-bottom:8px}}
.badge{{display:inline-block;background:#1e3a5f;border:1px solid #2563eb;border-radius:4px;
       padding:2px 10px;font-size:11px;color:#93c5fd;margin:2px 4px 2px 0}}
.badge.v{{background:#1e3d1e;border-color:#16a34a;color:#86efac}}
.badge.w{{background:#3a2a0a;border-color:#d97706;color:#fcd34d}}
.container{{max-width:1480px;margin:0 auto;padding:24px 48px}}
.st{{font-size:12px;color:#60a5fa;border-left:3px solid #2563eb;padding-left:12px;
    margin:28px 0 14px;letter-spacing:.8px;text-transform:uppercase;font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#1a2535;border:1px solid #253347;border-radius:8px;padding:16px;text-align:center}}
.card .lbl{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px}}
.card .val{{font-size:20px;font-weight:700}}
.pos{{color:#34d399}} .neg{{color:#f87171}} .neu{{color:#94a3b8}}
.row2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.row3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
.box{{background:#1a2535;border:1px solid #253347;border-radius:8px;padding:18px}}
.box h3{{font-size:11px;color:#64748b;margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px}}
.tw{{background:#1a2535;border:1px solid #253347;border-radius:8px;overflow:hidden;margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead{{background:#0c1220}}
th{{padding:9px 12px;text-align:right;color:#475569;font-weight:600;font-size:11px}}
th:first-child,th:nth-child(2),th:nth-child(3){{text-align:left}}
td{{padding:8px 12px;border-top:1px solid #0f1e30;text-align:right}}
td:first-child,td:nth-child(2),td:nth-child(3){{text-align:left}}
tr:hover{{background:#111e2e}}
.tag{{display:inline-block;padding:1px 7px;border-radius:3px;font-size:11px;white-space:nowrap}}
.tag.buy{{background:#1d3a7a;color:#93c5fd}}
.tag.rev-c{{background:#7a2c1d;color:#fca5a5}}
.tag.rev-o{{background:#7a4a1d;color:#fcd9a4}}
.tag.hdg{{background:#1d5a2c;color:#86efac}}
.tag.clr{{background:#1e293b;color:#64748b}}
.tag.c{{background:#1d3a7a;color:#93c5fd}}
.tag.p{{background:#4a1d7a;color:#ddd6fe}}
.tag.n{{background:#1e293b;color:#64748b}}
.note{{background:#0f2237;border:1px solid #1e3a5f;border-radius:8px;padding:14px 18px;margin-bottom:18px;font-size:13px;line-height:1.9}}
.note h4{{color:#60a5fa;margin-bottom:8px;font-size:14px}}
.note ul{{padding-left:20px;color:#94a3b8}}
.note li{{margin-bottom:3px}}
.note strong{{color:#e2e8f0}}
.roll-badge{{display:inline-block;background:#3a2a0a;border:1px solid #d97706;color:#fcd34d;
            border-radius:3px;padding:1px 6px;font-size:10px;margin-left:4px}}
.footer{{text-align:center;padding:20px;color:#334155;font-size:11px;
         border-top:1px solid #1a2535;margin-top:40px}}
</style>
</head>
<body>
<div class="hdr">
  <h1>创业板ETF (159915) 期权策略回测报告</h1>
  <div style="margin-bottom:10px">
    <span class="badge v">v3 动态换月</span>
    <span class="badge">2026年Q1</span>
    <span class="badge">1月→2月→3月→4月 滚动</span>
    <span class="badge">Black-76 实时重定价</span>
    <span class="badge w">距到期≤5交易日换月</span>
    <span class="badge">每30分钟验证</span>
    <span class="badge">对冲阈值5%</span>
  </div>
  <p style="color:#64748b;font-size:13px">
    1月用当月合约 → 1/21换2月合约 → 2/10换3月合约 → 3/18换4月合约 | 行权价按当日ETF价格动态生成(0.1间距)
  </p>
</div>

<div class="container">

<div class="note">
  <h4>v2 → v3 核心改进：动态合约换月</h4>
  <ul>
    <li><strong>v2问题</strong>：全程固定使用4月到期合约，1月交易时T≈90天，时间价值过高，与真实交易不符</li>
    <li><strong>v3改进</strong>：按换月规则动态选择合约 — 1月用1月合约(T≈20天)、2月用2月合约(T≈25天)等</li>
    <li><strong>换月节点</strong>：1/21(1月→2月)、2/10(2月→3月)、3/18(3月→4月)，共3次换月</li>
    <li><strong>IV调整</strong>：近月合约期限溢价更高（短期IV > 长期IV），反映真实市场波动结构</li>
    <li><strong>近月期权效果</strong>：T小 → Gamma大 → 价格对标的敏感度高 → 对冲收益更显著</li>
  </ul>
</div>

<div class="st">核心绩效指标</div>
<div class="cards">
  <div class="card"><div class="lbl">期末净值</div><div class="val neu">¥{perf['final_capital']:,.0f}</div></div>
  <div class="card"><div class="lbl">总收益率</div><div class="val {'pos' if perf['total_return']>0 else 'neg'}">{perf['total_return']:+.2f}%</div></div>
  <div class="card"><div class="lbl">年化收益率</div><div class="val {'pos' if perf['annual_return']>0 else 'neg'}">{perf['annual_return']:+.2f}%</div></div>
  <div class="card"><div class="lbl">夏普比率</div><div class="val pos">{perf['sharpe']:.3f}</div></div>
  <div class="card"><div class="lbl">最大回撤</div><div class="val neg">{perf['max_drawdown']:.2f}%</div></div>
  <div class="card"><div class="lbl">胜率</div><div class="val neu">{perf['win_rate']:.1f}%</div></div>
  <div class="card"><div class="lbl">盈亏比</div><div class="val pos">{perf['profit_loss_ratio']:.3f}</div></div>
  <div class="card"><div class="lbl">总交易笔数</div><div class="val neu">{perf['n_trades']}</div></div>
  <div class="card"><div class="lbl">开仓</div><div class="val neu">{perf['n_open']}</div></div>
  <div class="card"><div class="lbl">日内反手</div><div class="val neg">{perf['n_reverse']}</div></div>
  <div class="card"><div class="lbl">对冲开仓</div><div class="val pos">{perf['n_hedge']}</div></div>
  <div class="card"><div class="lbl">换月次数</div><div class="val neu">3</div></div>
</div>

<div class="row2">
  <div class="box">
    <h3>策略累计净值（换月节点标注）</h3>
    <canvas id="navC" height="220"></canvas>
  </div>
  <div class="box">
    <h3>ETF 159915 收盘价</h3>
    <canvas id="etfC" height="220"></canvas>
  </div>
</div>

<div class="row3">
  <div class="box">
    <h3>每日盈亏 (¥)</h3>
    <canvas id="pnlC" height="200"></canvas>
  </div>
  <div class="box">
    <h3>回撤曲线 (%)</h3>
    <canvas id="ddC"  height="200"></canvas>
  </div>
  <div class="box">
    <h3>月度盈亏 & 胜率</h3>
    <canvas id="mnthC" height="200"></canvas>
  </div>
</div>

<div class="row2">
  <div class="box">
    <h3>反手时间分布（日内节点）</h3>
    <canvas id="revC" height="180"></canvas>
  </div>
  <div class="box">
    <h3>对冲触发时间分布（日内节点）</h3>
    <canvas id="hdgC" height="180"></canvas>
  </div>
</div>

<div class="st">合约换月时间线</div>
<div class="tw">
<table>
  <thead><tr>
    <th>日期</th><th>使用合约</th><th>ETF开盘</th><th>ETF收盘</th>
    <th>信号</th><th>反手</th><th>对冲</th><th>当日盈亏</th><th>累计净值</th>
  </tr></thead>
  <tbody>
"""
prev_roll = None
for _, r in daily.iterrows():
    curr_roll = str(r["roll"]) if pd.notna(r.get("roll")) else ""
    is_roll   = prev_roll is not None and curr_roll != prev_roll
    prev_roll = curr_roll

    pnl_c  = "pos" if r["daily_pnl"]>0 else ("neg" if r["daily_pnl"]<0 else "neu")
    sig = r["signal"]
    stag = ('<span class="tag c">认购↑</span>' if sig=="认购" else
            '<span class="tag p">认沽↓</span>' if sig=="认沽" else
            '<span class="tag n">观望</span>')
    rev_tag = '<span class="tag rev-c">✓反手</span>' if r["reversed"] else ''
    hdg_tag = '<span class="tag hdg">✓对冲</span>' if r["hedged"]   else ''
    roll_tag= '<span class="roll-badge">换月</span>' if is_roll else ''
    label   = r["expiry_label"]
    html   += f"""
    <tr>
      <td>{r['date']}</td>
      <td><span style="color:#94a3b8">{label}</span>{roll_tag}</td>
      <td>{r['etf_open']:.3f}</td><td>{r['etf_close']:.3f}</td>
      <td>{stag}</td><td>{rev_tag}</td><td>{hdg_tag}</td>
      <td class="{pnl_c}">¥{r['daily_pnl']:,.2f}</td>
      <td>¥{r['total_value']:,.2f}</td>
    </tr>"""
html += "</tbody></table></div>"

html += """
<div class="st">月度绩效</div>
<div class="tw"><table>
  <thead><tr>
    <th>月份</th><th>交易日</th><th>月盈亏</th><th>胜率</th>
    <th>最大盈利/天</th><th>最大亏损/天</th><th>反手次数</th><th>对冲次数</th><th>月末净值</th>
  </tr></thead><tbody>
"""
for _, r in monthly.iterrows():
    c = "pos" if r["月盈亏"]>0 else "neg"
    html += f"""<tr>
      <td>{r['month']}</td><td>{int(r['天数'])}</td>
      <td class="{c}">¥{r['月盈亏']:,.2f}</td>
      <td>{float(r['胜率'])*100:.1f}%</td>
      <td class="pos">¥{r['最大盈利']:,.2f}</td>
      <td class="neg">¥{r['最大亏损']:,.2f}</td>
      <td>{int(r['反手次数'])}</td><td>{int(r['对冲次数'])}</td>
      <td>¥{r['月末净值']:,.2f}</td>
    </tr>"""
html += "</tbody></table></div>"

html += """
<div class="st">完整交易记录</div>
<div class="tw"><table>
  <thead><tr>
    <th>日期</th><th>时间</th><th>动作</th><th>期权名称</th>
    <th>方向</th><th>行权价</th><th>到期日</th>
    <th>权利金</th><th>合约价值</th><th>盈亏</th><th>原因</th>
  </tr></thead><tbody>
"""
for _, r in trades.iterrows():
    act = r["action"]
    amap = {"买入开仓": "buy", "反手平仓":"rev-c","反手开仓":"rev-o",
            "卖出对冲":"hdg","卖出平仓":"clr","买入平仓":"clr"}
    acls  = amap.get(act,"clr")
    atag  = f'<span class="tag {acls}">{act}</span>'
    pnl   = float(r["pnl"]) if not pd.isna(r["pnl"]) else 0
    pnl_c = "pos" if pnl>0 else ("neg" if pnl<0 else "")
    html += f"""<tr>
      <td>{r['date']}</td><td>{r['time']}</td>
      <td>{atag}</td>
      <td style="white-space:nowrap">{r['option_name']}</td>
      <td>{r['direction']}</td>
      <td>{r['strike']:.1f}</td>
      <td style="color:#64748b;font-size:11px">{r['expiry']}</td>
      <td>{float(r['price']):.4f}</td>
      <td>¥{float(r['premium_value']):,.0f}</td>
      <td class="{pnl_c}">{'¥'+f'{pnl:,.2f}' if pnl!=0 else '—'}</td>
      <td style="color:#64748b;font-size:11px">{r['reason']}</td>
    </tr>"""
html += "</tbody></table></div>"

roll_data  = json.dumps([d for d in roll_days])
rev_labels = json.dumps(list(rev.index))
rev_vals   = json.dumps(list(map(int,rev.values)))
hdg_labels = json.dumps(list(hdg.index))
hdg_vals   = json.dumps(list(map(int,hdg.values)))

html += f"""
<div class="footer">
  创业板ETF(159915) 期权三阶段策略 v3（动态换月）| Black-76 实时重定价 | IV期限结构调整 |
  数据：AKShare | 数据库：VeighNA SQLite | 策略参照：etf-option-strategy
</div>

<script>
const gc='rgba(37,51,71,0.6)',tc='#64748b';
const navDates={json.dumps(dates_js)},navVals={json.dumps(nav_js)};
const pnlVals={json.dumps(pnl_js)},ddVals={json.dumps(dd_js)};
const etfD={json.dumps(etf_d_js)},etfC={json.dumps(etf_c_js)};
const rollDays={roll_data};
const mnthL={json.dumps(m_labels)},mnthP={json.dumps(m_pnl)},mnthW={json.dumps(m_win)};
const revL={rev_labels},revV={rev_vals},hdgL={hdg_labels},hdgV={hdg_vals};

const baseOpts=(extra={{}})=>Object.assign({{
  responsive:true,
  plugins:{{legend:{{labels:{{color:tc,boxWidth:10,font:{{size:11}}}}}}}},
  scales:{{
    x:{{ticks:{{color:tc,maxTicksLimit:8,font:{{size:10}}}},grid:{{color:gc}}}},
    y:{{ticks:{{color:tc,font:{{size:10}}}},grid:{{color:gc}}}}
  }}
}},extra);

// 换月垂直线 plugin
const rollPlugin={{
  id:'rollLines',
  afterDraw(chart){{
    rollDays.forEach(d=>{{
      const idx=navDates.indexOf(d);
      if(idx<0)return;
      const {{ctx,chartArea,scales}}=chart;
      const x=scales.x.getPixelForValue(idx);
      ctx.save();
      ctx.strokeStyle='rgba(251,191,36,0.45)';
      ctx.lineWidth=1.5;
      ctx.setLineDash([4,3]);
      ctx.beginPath();ctx.moveTo(x,chartArea.top);ctx.lineTo(x,chartArea.bottom);ctx.stroke();
      ctx.fillStyle='#fbbf24';ctx.font='10px sans-serif';ctx.textAlign='center';
      ctx.fillText('换月',x,chartArea.top+10);
      ctx.restore();
    }});
  }}
}};

new Chart(document.getElementById('navC'),{{
  type:'line',
  data:{{labels:navDates,datasets:[
    {{label:'策略净值',data:navVals,borderColor:'#60a5fa',
     backgroundColor:'rgba(96,165,250,0.06)',borderWidth:2,pointRadius:0,fill:true,tension:0.2}},
    {{label:'基准1.0',data:navDates.map(()=>1),borderColor:'#334155',
     borderDash:[4,4],borderWidth:1,pointRadius:0}}
  ]}},
  options:baseOpts(),
  plugins:[rollPlugin]
}});

new Chart(document.getElementById('etfC'),{{type:'line',data:{{labels:etfD,datasets:[
  {{label:'159915',data:etfC,borderColor:'#a78bfa',
   backgroundColor:'rgba(167,139,250,0.06)',borderWidth:2,pointRadius:0,fill:true,tension:0.2}}
]}},options:baseOpts()}});

new Chart(document.getElementById('pnlC'),{{type:'bar',data:{{labels:navDates,datasets:[
  {{label:'日盈亏(¥)',data:pnlVals,
   backgroundColor:pnlVals.map(v=>v>0?'rgba(52,211,153,0.7)':'rgba(248,113,113,0.7)'),
   borderColor:pnlVals.map(v=>v>0?'#34d399':'#f87171'),borderWidth:1}}
]}},options:baseOpts({{plugins:{{legend:{{display:false}}}}}})}});

new Chart(document.getElementById('ddC'),{{type:'line',data:{{labels:navDates,datasets:[
  {{label:'回撤%',data:ddVals,borderColor:'#f87171',
   backgroundColor:'rgba(248,113,113,0.08)',borderWidth:2,pointRadius:0,fill:true,tension:0.2}}
]}},options:baseOpts()}});

new Chart(document.getElementById('mnthC'),{{
  type:'bar',
  data:{{labels:mnthL,datasets:[
    {{label:'月盈亏(¥)',data:mnthP,backgroundColor:mnthP.map(v=>v>0?'rgba(52,211,153,0.7)':'rgba(248,113,113,0.7)'),
     yAxisID:'y',order:1}},
    {{label:'胜率(%)',data:mnthW,type:'line',borderColor:'#fbbf24',borderWidth:2,
     pointRadius:5,yAxisID:'y1',order:0}}
  ]}},
  options:{{
    responsive:true,
    plugins:{{legend:{{labels:{{color:tc,font:{{size:11}}}}}}}},
    scales:{{
      x:{{ticks:{{color:tc}},grid:{{color:gc}}}},
      y:{{ticks:{{color:tc}},grid:{{color:gc}},title:{{display:true,text:'¥',color:tc}}}},
      y1:{{ticks:{{color:'#fbbf24'}},grid:{{display:false}},
           position:'right',title:{{display:true,text:'胜率%',color:'#fbbf24'}}}}
    }}
  }}
}});

new Chart(document.getElementById('revC'),{{type:'bar',data:{{labels:revL,datasets:[
  {{label:'反手平仓次数',data:revV,backgroundColor:'rgba(251,146,60,0.7)',borderColor:'#fb923c',borderWidth:1}}
]}},options:baseOpts({{plugins:{{legend:{{display:false}}}}}})}});

new Chart(document.getElementById('hdgC'),{{type:'bar',data:{{labels:hdgL,datasets:[
  {{label:'对冲触发次数',data:hdgV,backgroundColor:'rgba(52,211,153,0.7)',borderColor:'#34d399',borderWidth:1}}
]}},options:baseOpts({{plugins:{{legend:{{display:false}}}}}})}});
</script>
</body></html>"""

out = f"{D}/backtest_report_v3.html"
with open(out,"w",encoding="utf-8") as f:
    f.write(html)
print(f"✓ v3 报告已生成: {out}  ({os.path.getsize(out)/1024:.1f} KB)")
