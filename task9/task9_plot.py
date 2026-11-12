"""
Task 9 — Risk-return diagram
Run from the extracted task9_verified directory:
    python task9_plot.py
Required packages: numpy, pandas, matplotlib
All input CSV files are in ./inputs. Outputs are saved to ./outputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INP = ROOT / "inputs"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ASSETS = ["MSCI US","MMM","GOOGL","AXP","AAPL","CVX","CSCO","HON","TRV","V","WMT"]

cov = pd.read_csv(INP / "task3_covariance_matrix(1).csv").set_index("Asset")
Sigma = cov.loc[ASSETS, ASSETS].to_numpy(float)
returns = pd.read_csv(INP / "task4_expected_excess_returns.csv").set_index("Asset")
mu = returns.loc[ASSETS, "Expected_Excess_Return_pct_pa"].to_numpy(float) / 100

front = pd.read_csv(INP / "task3_frontier_21_portfolios(1).csv")
front_r = front["Actual_Excess_Return_pct"].to_numpy(float)
front_sd = np.sqrt(front["Variance"].to_numpy(float)) * 100

gmvp = pd.read_csv(INP / "task3_gmvp_weights(1).csv").set_index("Asset")
w_g = gmvp.loc[ASSETS, "Weight"].to_numpy(float)
gmvp_y = w_g @ mu * 100
gmvp_x = np.sqrt(w_g @ Sigma @ w_g) * 100

tan_w = pd.read_csv(INP / "task5_optimal_weights(1).csv").set_index("Asset")
w_t = tan_w.loc[ASSETS, "Weight"].to_numpy(float)
summary = pd.read_csv(INP / "task5_portfolio_summary(1).csv").set_index("Metric")["Value"]
tan_y = float(summary["Expected_excess_return"]) * 100
tan_x = float(summary["Risk_volatility"]) * 100
sharpe = float(summary["Sharpe_ratio"])

cp = pd.read_csv(INP / "task7_complete_portfolio_weights.csv")
w_c = cp[cp.Asset.isin(ASSETS)].set_index("Asset").loc[ASSETS, "Complete_Portfolio_Weight"].to_numpy(float)
s7 = pd.read_csv(INP / "task7_summary.csv").set_index("Metric")
complete_y = float(s7.loc["Expected excess return", "Decimal"]) * 100
complete_x = float(s7.loc["Annualized risk (standard deviation)", "Decimal"]) * 100

asset_x = np.sqrt(np.diag(Sigma)) * 100
asset_y = mu * 100
gmvp_idx = int(np.argmin(np.abs(front_r - gmvp_y)))

fig, ax = plt.subplots(figsize=(12.5, 8))
ax.plot(front_sd[:gmvp_idx+1], front_r[:gmvp_idx+1], "--o",
        linewidth=1.8, markersize=4, label="Lower (inefficient) frontier branch")
ax.plot([gmvp_x, *front_sd[gmvp_idx+1:]], [gmvp_y, *front_r[gmvp_idx+1:]],
        "-o", linewidth=2.5, markersize=4, label="Efficient frontier")
ax.scatter(asset_x, asset_y, marker="^", s=75, edgecolors="black",
           linewidths=.5, label="Individual assets", zorder=4)
for x, y, name in zip(asset_x, asset_y, ASSETS):
    ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

ax.scatter([gmvp_x], [gmvp_y], marker="D", s=105, label="GMVP", zorder=6)
ax.annotate("GMVP", (gmvp_x, gmvp_y), xytext=(-42,-17), textcoords="offset points")
ax.scatter([tan_x], [tan_y], marker="*", s=240, label="Tangency portfolio", zorder=7)
ax.annotate("Tangency", (tan_x,tan_y), xytext=(8,8), textcoords="offset points")
ax.scatter([complete_x], [complete_y], marker="P", s=125,
           label="Complete portfolio (A=6)", zorder=7)
ax.annotate("Complete portfolio (A=6)", (complete_x,complete_y),
            xytext=(8,-17), textcoords="offset points")
ax.scatter([0],[0], marker="o", s=55, label="Risk-free asset", zorder=6)
xmax = max(asset_x.max(), front_sd.max(), tan_x) * 1.06
xx = np.linspace(0, xmax, 200)
ax.plot(xx, sharpe*xx, "-.", linewidth=2, label=f"CAL (Sharpe = {sharpe:.4f})")
ax.set_xlabel("Annualized risk (standard deviation, % p.a.)")
ax.set_ylabel("Expected excess return (% p.a.)")
ax.set_title("Task 9 — Risk–Return Diagram")
ax.grid(True, alpha=.28)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "task9_risk_return_diagram.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", OUT / "task9_risk_return_diagram.png")
