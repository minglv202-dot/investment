import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "inputs" / "task8_inputs.json").read_text(encoding="utf-8"))
mu = float(data["tangency_expected_excess_return_annual"])
var = float(data["tangency_variance_annual"])
sigma = var ** 0.5
acrit = mu / var

out = ROOT / "outputs"
out.mkdir(exist_ok=True)
with (out / "task8_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Value", "Interpretation"])
    w.writerow(["Critical risk aversion parameter", f"{acrit:.10f}", "Leverage if A is below this value"])
    w.writerow(["Tangency expected excess return", f"{mu:.10f}", f"{mu*100:.6f}% p.a."])
    w.writerow(["Tangency variance", f"{var:.10f}", "Annual variance"])
    w.writerow(["Tangency standard deviation", f"{sigma:.10f}", f"{sigma*100:.6f}% p.a."])
    w.writerow(["Leverage condition", f"A < {acrit:.10f}", "Optimal tangency allocation y* > 1"])
    w.writerow(["Boundary", f"A = {acrit:.10f}", "100% invested in tangency portfolio"])
    w.writerow(["Lending condition", f"A > {acrit:.10f}", "Optimal tangency allocation y* < 1"])

with (out / "task8_examples.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Risk_aversion_A", "Optimal_tangency_allocation_y", "Risk_free_allocation", "Decision"])
    for A in [1.5, acrit, 4.0, 6.0]:
        y = acrit / A
        decision = "Leverage" if y > 1 + 1e-10 else ("100% tangency" if abs(y-1) <= 1e-10 else "Lending")
        w.writerow([f"{A:.10f}", f"{y:.10f}", f"{1-y:.10f}", decision])

print(f"Critical risk aversion parameter: {acrit:.10f}")
print(f"Leverage if A < {acrit:.10f}")
print(f"100% tangency if A = {acrit:.10f}")
print(f"Lending if A > {acrit:.10f}")
