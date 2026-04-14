# -*- coding: utf-8 -*-
"""
Heston模型多维投资决策可视化看板
使用 Plotly 生成交互式 HTML 及高分辨率 PNG
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import pearsonr

# ===================== 数据读取与预处理 =====================
# 读取实际股价数据 (A5:D134)
df_raw = pd.read_excel("../data/stock_price.xlsx", sheet_name="Sheet1", header=None)
df = df_raw.iloc[4:134].copy()
df.columns = ["Col0", "Date", "Open", "Close"]
df = df[["Date", "Open", "Close"]].reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"])
df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna().reset_index(drop=True)

# 计算均价、对数收益率
S = (df["Open"] + df["Close"]) / 2
df["AvgPrice"] = S
log_S = np.log(S)
R = np.diff(log_S)
R = np.insert(R, 0, 0)
df["LogReturn"] = R

# 移动平均线
df["MA5"] = df["AvgPrice"].rolling(window=5, min_periods=1).mean()
df["MA10"] = df["AvgPrice"].rolling(window=10, min_periods=1).mean()
df["MA20"] = df["AvgPrice"].rolling(window=20, min_periods=1).mean()

# 滚动波动率 (10日窗口)
W = 10
n_windows = len(R) - W + 1
d = np.array([np.std(R[i : i + W], ddof=1) for i in range(n_windows)])

# Heston参数
O = np.mean(d)
d_v = np.std(d, ddof=1)
d_prev = d[:-1]
delta_d = np.diff(d)
centered_d_prev = d_prev - O
n = len(delta_d)
cov_val = np.sum(centered_d_prev * delta_d) / (n - 1)
var_val = np.sum(centered_d_prev ** 2) / (n - 1)
K = -cov_val / var_val

length_for_corr = len(d)
A = R[:length_for_corr]
p_corr, _ = pearsonr(A, d)

# 波动率对齐
df["RollingVol"] = np.nan
df.loc[W - 1 :, "RollingVol"] = d

# 累计收益与回撤
df["CumReturn"] = np.cumsum(df["LogReturn"])
rolling_max = df["CumReturn"].cummax()
df["Drawdown"] = df["CumReturn"] - rolling_max

# 年化指标
annual_return = df["LogReturn"].mean() * 252
annual_vol = df["LogReturn"].std() * np.sqrt(252)
sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
var_95 = np.percentile(df["LogReturn"], 5)
var_99 = np.percentile(df["LogReturn"], 1)
max_dd = df["Drawdown"].min()

# 读取Heston模拟路径
sim_df = pd.read_csv("../results/heston_simulation_paths.csv", index_col=0)
sim_df.index = sim_df.index.astype(float)
N_paths = sim_df.shape[1]

# ===================== Plotly 看板构建 =====================
fig = make_subplots(
    rows=4,
    cols=2,
    subplot_titles=(
        "① 实际股价走势与移动平均线",
        "② Heston模型模拟路径分布",
        "③ 对数收益率与滚动波动率",
        "④ 收益率分布与正态拟合",
        "⑤ 收益率-波动率相关性分析",
        "⑥ 累计收益与最大回撤",
        "⑦ 模拟路径期末价格分布",
        "⑧ 关键投资指标与Heston参数",
    ),
    specs=[
        [{}, {}],
        [{}, {}],
        [{}, {}],
        [{}, {"type": "table"}],
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.10,
)

# 对齐长度
align_len = min(100, len(df))
align_dates = df["Date"].iloc[:align_len]

# ---- ① 实际股价走势 ----
fig.add_trace(go.Scatter(x=df["Date"], y=df["Open"], mode="lines", name="开盘价", line=dict(color="#1f77b4", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="收盘价", line=dict(color="#ff7f0e", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=df["AvgPrice"], mode="lines", name="均价", line=dict(color="#2ca02c", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA5"], mode="lines", name="MA5", line=dict(color="#d62728", width=1, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA10"], mode="lines", name="MA10", line=dict(color="#9467bd", width=1, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20", line=dict(color="#8c564b", width=1, dash="dash")), row=1, col=1)

# ---- ② Heston模拟路径 ----
for col in sim_df.columns:
    fig.add_trace(
        go.Scatter(
            x=align_dates,
            y=sim_df[col].iloc[:align_len].values,
            mode="lines",
            line=dict(width=0.8, color="gray"),
            showlegend=False,
            opacity=0.5,
            hoverinfo="skip",
            name=col,
        ),
        row=1,
        col=2,
    )
sim_mean = sim_df.iloc[:align_len].mean(axis=1)
sim_p5 = sim_df.iloc[:align_len].quantile(0.05, axis=1)
sim_p95 = sim_df.iloc[:align_len].quantile(0.95, axis=1)
fig.add_trace(go.Scatter(x=align_dates, y=sim_mean, mode="lines", name="模拟均值", line=dict(color="red", width=2)), row=1, col=2)
fig.add_trace(go.Scatter(x=align_dates, y=sim_p95, mode="lines", name="95%分位", line=dict(color="red", width=1, dash="dot"), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=align_dates, y=sim_p5, mode="lines", name="5%分位", line=dict(color="red", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(255,0,0,0.1)", showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=align_dates, y=df["AvgPrice"].iloc[:align_len], mode="lines", name="实际均价", line=dict(color="black", width=2)), row=1, col=2)

# ---- ③ 对数收益率与滚动波动率 ----
colors_bar = ["green" if x >= 0 else "red" for x in df["LogReturn"]]
fig.add_trace(go.Bar(x=df["Date"], y=df["LogReturn"], name="日对数收益率", marker_color=colors_bar, opacity=0.7, showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=df["Date"], y=df["RollingVol"], mode="lines", name="10日滚动波动率", line=dict(color="crimson", width=2)), row=2, col=1)
fig.add_hline(y=O, line_dash="dash", line_color="green", annotation_text=f"长期均值 θ={O:.4f}", row=2, col=1)

# ---- ④ 收益率分布 ----
valid_r = R[R != 0]
hist, bins = np.histogram(valid_r, bins=30, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2
fig.add_trace(go.Bar(x=bin_centers, y=hist, name="收益率直方图", marker_color="lightblue", showlegend=False), row=2, col=2)
mu_r, std_r = stats.norm.fit(valid_r)
x_norm = np.linspace(valid_r.min(), valid_r.max(), 100)
fig.add_trace(go.Scatter(x=x_norm, y=stats.norm.pdf(x_norm, mu_r, std_r), mode="lines", name="正态拟合", line=dict(color="red", width=2)), row=2, col=2)

# ---- ⑤ 收益率-波动率散点 ----
valid_idx = ~np.isnan(df["RollingVol"])
x_scat = df.loc[valid_idx, "LogReturn"]
y_scat = df.loc[valid_idx, "RollingVol"]
fig.add_trace(go.Scatter(x=x_scat, y=y_scat, mode="markers", name="收益率-波动率", marker=dict(color="navy", size=6, opacity=0.6), showlegend=False), row=3, col=1)
z = np.polyfit(x_scat, y_scat, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(x_scat.min(), x_scat.max(), 100)
fig.add_trace(go.Scatter(x=x_line, y=p_fit(x_line), mode="lines", name="线性拟合", line=dict(color="red", width=2)), row=3, col=1)

# ---- ⑥ 累计收益与回撤 ----
fig.add_trace(go.Scatter(x=df["Date"], y=df["CumReturn"], mode="lines", name="累计对数收益", line=dict(color="green", width=2), fill="tozeroy", fillcolor="rgba(0,128,0,0.1)"), row=3, col=2)
fig.add_trace(go.Scatter(x=df["Date"], y=df["Drawdown"], mode="lines", name="回撤", line=dict(color="red", width=2), fill="tozeroy", fillcolor="rgba(255,0,0,0.2)"), row=3, col=2)

# ---- ⑦ 期末价格分布 ----
final_prices = sim_df.iloc[-1].values
fig.add_trace(go.Histogram(x=final_prices, nbinsx=15, name="期末价格分布", marker_color="coral", opacity=0.75, showlegend=False), row=4, col=1)
fig.add_vline(x=final_prices.mean(), line_dash="dash", line_color="darkred", annotation_text=f"均值={final_prices.mean():.2f}", row=4, col=1)

# ---- ⑧ 参数表格 ----
table_data = dict(
    values=[
        ["年化收益率", "年化波动率", "夏普比率", "日VaR(95%)", "日VaR(99%)", "最大回撤"],
        [f"{annual_return:.2%}", f"{annual_vol:.2%}", f"{sharpe:.3f}", f"{var_95:.4f}", f"{var_99:.4f}", f"{max_dd:.4f}"],
    ]
)
fig.add_trace(
    go.Table(
        header=dict(values=["<b>投资指标</b>", "<b>数值</b>"], fill_color="paleturquoise", align="center", font=dict(size=14)),
        cells=dict(values=table_data["values"], align="center", fill_color="white", font=dict(size=13)),
    ),
    row=4,
    col=2,
)

# 布局
fig.update_layout(
    title=dict(text="<b>邮储银行 (601658.SH) Heston模型投资决策多维看板</b>", x=0.5, font_size=20),
    height=1400,
    width=1600,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    hovermode="x unified",
    template="plotly_white",
)

fig.update_yaxes(title_text="价格 (CNY)", row=1, col=1)
fig.update_yaxes(title_text="价格 (CNY)", row=1, col=2)
fig.update_yaxes(title_text="对数收益率", row=2, col=1)
fig.update_yaxes(title_text="概率密度", row=2, col=2)
fig.update_yaxes(title_text="滚动波动率", row=3, col=1)
fig.update_yaxes(title_text="累计收益 / 回撤", row=3, col=2)
fig.update_yaxes(title_text="频数", row=4, col=1)

# 保存输出
fig.write_html("../results/investment_dashboard.html", include_plotlyjs="cdn")
print("交互式看板已保存: investment_dashboard.html")

fig.write_image("../results/investment_dashboard.png", width=1600, height=1400, scale=2)
print("静态看板 PNG 已保存: investment_dashboard.png")

# 同时输出一个补充的参数汇总文本文件
with open("../results/dashboard_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("邮储银行 (601658.SH) Heston模型投资决策分析汇总\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"数据区间: {df['Date'].min().date()} ~ {df['Date'].max().date()}\n")
    f.write(f"交易日数量: {len(df)}\n")
    f.write(f"模拟路径数量: {N_paths}\n\n")
    f.write("[Heston 模型参数]\n")
    f.write(f"  均值回归速度 (κ)         = {K:.6f}\n")
    f.write(f"  波动率长期均值 (θ)       = {O:.6f}\n")
    f.write(f"  波动率的波动率 (σ_v)     = {d_v:.6f}\n")
    f.write(f"  收益率-波动率相关系数 (ρ)= {p_corr:.6f}\n")
    f.write(f"  初始波动率 (v0)          = {O:.6f}\n\n")
    f.write("[投资指标]\n")
    f.write(f"  年化收益率               = {annual_return:.2%}\n")
    f.write(f"  年化波动率               = {annual_vol:.2%}\n")
    f.write(f"  夏普比率                 = {sharpe:.3f}\n")
    f.write(f"  日VaR (95%)              = {var_95:.4f}\n")
    f.write(f"  日VaR (99%)              = {var_99:.4f}\n")
    f.write(f"  最大回撤                 = {max_dd:.4f}\n\n")
    f.write("[模拟期末价格统计]\n")
    f.write(f"  均值                     = {final_prices.mean():.4f}\n")
    f.write(f"  标准差                   = {final_prices.std():.4f}\n")
    f.write(f"  最小值                   = {final_prices.min():.4f}\n")
    f.write(f"  最大值                   = {final_prices.max():.4f}\n")
    f.write(f"  中位数                   = {np.median(final_prices):.4f}\n")
    f.write("\n" + "=" * 60 + "\n")

print("分析汇总已保存: dashboard_summary.txt")
print("\n========== 分析完成 ==========")
print(f"数据区间: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"Heston参数: κ={K:.4f}, θ={O:.4f}, σ_v={d_v:.4f}, ρ={p_corr:.4f}")
print(f"风险指标: 年化收益={annual_return:.2%}, 年化波动={annual_vol:.2%}, 夏普={sharpe:.3f}, 最大回撤={max_dd:.4f}")
