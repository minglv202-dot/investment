"""
Task 6 — Capital Allocation Line (CAL)
Uses the long-only tangency portfolio obtained in Task 5.
All return and risk inputs are annual and expressed in decimal units.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

portfolio = pd.read_csv(INPUTS / "task5_portfolio_summary.csv").iloc[0]
weights = pd.read_csv(INPUTS / "task5_optimal_weights.csv")

mu_t = float(portfolio["Expected_Excess_Return"])
variance_t = float(portfolio["Annual_Variance"])
sigma_t = float(portfolio["Annual_Risk"])
rf = float(portfolio["Risk_Free_Rate"])
sharpe = mu_t / sigma_t

# Validate inherited Task 5 inputs.
assert np.isclose(sigma_t ** 2, variance_t, atol=1e-9)
assert np.isclose(weights["Weight"].sum(), 1.0, atol=1e-7)
assert (weights["Weight"] >= -1e-10).all()
assert np.isclose(sharpe, float(portfolio["Sharpe_Ratio"]), atol=1e-7)

# y is the fraction of total wealth invested in the tangency portfolio.
# y > 1 represents borrowing at the risk-free rate (leverage).
y_values = np.linspace(0.0, 1.5, 151)
cal = pd.DataFrame({
    "Tangency_Allocation": y_values,
    "Risk_Free_Allocation": 1.0 - y_values,
    "Annual_Risk_Decimal": y_values * sigma_t,
    "Expected_Excess_Return_Decimal": y_values * mu_t,
})
cal["Annual_Risk_Percent"] = 100 * cal["Annual_Risk_Decimal"]
cal["Expected_Excess_Return_Percent"] = 100 * cal["Expected_Excess_Return_Decimal"]
cal.to_csv(OUTPUTS / "task6_cal_coordinates.csv", index=False)

summary = pd.DataFrame([
    ["Risk-free rate (% p.a.)", 100 * rf],
    ["Tangency expected excess return (% p.a.)", 100 * mu_t],
    ["Tangency annual variance", variance_t],
    ["Tangency annual risk (% p.a.)", 100 * sigma_t],
    ["CAL slope / Sharpe ratio", sharpe],
])
summary.columns = ["Metric", "Value"]
summary.to_csv(OUTPUTS / "task6_cal_summary.csv", index=False)

# Plot in the expected excess return–risk plane.
plt.figure(figsize=(9, 6))
plt.plot(cal["Annual_Risk_Percent"], cal["Expected_Excess_Return_Percent"],
         label=f"CAL (slope = {sharpe:.4f})")
plt.scatter([0], [0], label="Risk-free asset")
plt.scatter([100 * sigma_t], [100 * mu_t], marker="*", s=180,
            label="Tangency portfolio")
plt.annotate("Risk-free asset (0, 0)", (0, 0), xytext=(8, 8),
             textcoords="offset points")
plt.annotate(f"Tangency portfolio ({100*sigma_t:.4f}%, {100*mu_t:.4f}%)",
             (100*sigma_t, 100*mu_t), xytext=(10, -25),
             textcoords="offset points")
plt.xlabel("Annual Risk (Standard Deviation, %)")
plt.ylabel("Expected Annual Excess Return (%)")
plt.title("Capital Allocation Line — Task 6")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "task6_cal.png", dpi=300)
plt.show()

print(f"Sharpe ratio / CAL slope: {sharpe:.8f}")
print(f"CAL: E(R_C)-Rf = {sharpe:.8f} * sigma_C")
print(f"Tangency point: risk={100*sigma_t:.6f}%, "
      f"excess return={100*mu_t:.6f}%")
