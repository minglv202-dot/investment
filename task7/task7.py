"""
Task 7 - Optimal Complete Portfolio
Reproduces the complete-portfolio allocation using Task 5 tangency-portfolio inputs.
All calculations use decimal returns and unrounded inputs.
"""
import json
import math
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
with (ROOT / "inputs" / "task7_inputs.json").open(encoding="utf-8") as f:
    data = json.load(f)

A = float(data["risk_aversion_A"])
rf = float(data["risk_free_rate_annual"])
mu_t = float(data["tangency_expected_excess_return_annual"])
var_t = float(data["tangency_variance_annual"])
assets = list(data["tangency_weights"].keys())
w_t = [float(data["tangency_weights"][a]) for a in assets]

# Optimal fraction invested in the tangency portfolio
y = mu_t / (A * var_t)
w_rf = 1.0 - y

# Complete-portfolio moments
expected_excess_return = y * mu_t
expected_total_return = rf + expected_excess_return
variance = y**2 * var_t
risk = math.sqrt(variance)
sharpe = expected_excess_return / risk

# Complete-portfolio composition
w_complete = [y * wi for wi in w_t]
assert abs(sum(w_complete) + w_rf - 1.0) < 1e-9
assert 0.0 <= y <= 1.0, "This result assumes a non-leveraged allocation."

out = ROOT / "outputs"
out.mkdir(exist_ok=True)

with (out / "task7_complete_portfolio_weights.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Asset", "Tangency_Portfolio_Weight", "Complete_Portfolio_Weight", "Complete_Portfolio_Weight_Percent"])
    for asset, wt, wc in zip(assets, w_t, w_complete):
        writer.writerow([asset, f"{wt:.10f}", f"{wc:.10f}", f"{wc*100:.6f}"])
    writer.writerow(["Risk-free asset", "", f"{w_rf:.10f}", f"{w_rf*100:.6f}"])
    writer.writerow(["TOTAL", "", f"{sum(w_complete)+w_rf:.10f}", f"{(sum(w_complete)+w_rf)*100:.6f}"])

with (out / "task7_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Decimal", "Percent_or_value"])
    rows = [
        ("Optimal tangency allocation y*", y, f"{y*100:.6f}%"),
        ("Risk-free allocation", w_rf, f"{w_rf*100:.6f}%"),
        ("Expected excess return", expected_excess_return, f"{expected_excess_return*100:.6f}%"),
        ("Expected total return", expected_total_return, f"{expected_total_return*100:.6f}%"),
        ("Variance", variance, ""),
        ("Annualized risk (standard deviation)", risk, f"{risk*100:.6f}%"),
        ("Sharpe ratio", sharpe, ""),
    ]
    for name, value, display in rows:
        writer.writerow([name, f"{value:.12f}", display])

print(f"Optimal tangency allocation y*: {y:.10%}")
print(f"Risk-free allocation: {w_rf:.10%}")
print(f"Expected excess return: {expected_excess_return:.10%}")
print(f"Expected total return: {expected_total_return:.10%}")
print(f"Annualized risk: {risk:.10%}")
print(f"Variance: {variance:.12f}")
print(f"Sharpe ratio: {sharpe:.10f}")
print(f"Weight sum: {sum(w_complete)+w_rf:.12f}")
