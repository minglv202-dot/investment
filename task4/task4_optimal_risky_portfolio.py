"""
Task 4 — Optimal Risky Portfolio
Maximize the Sharpe ratio subject to fully-invested and long-only constraints.

Inputs:
  task4_expected_excess_returns.csv
  task4_covariance_matrix.csv

Run from this folder:
  python task4_optimal_risky_portfolio.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MU_FILE = "task4_expected_excess_returns.csv"
COV_FILE = "task4_covariance_matrix.csv"
WEIGHTS_FILE = "task4_optimal_weights.csv"
SUMMARY_FILE = "task4_portfolio_summary.csv"

# 1. Load annual expected excess returns and annual covariance matrix.
mu_df = pd.read_csv(MU_FILE)
assets = mu_df["Asset"].tolist()
mu = mu_df["Expected_Excess_Return_pct_pa"].to_numpy(dtype=float) / 100.0

cov_df = pd.read_csv(COV_FILE, index_col="Asset")
Sigma = cov_df.loc[assets, assets].to_numpy(dtype=float)

# 2. Define the negative Sharpe ratio.
#    scipy.optimize.minimize minimizes, so maximize Sharpe by minimizing -Sharpe.
def negative_sharpe(w):
    excess_return = w @ mu
    variance = w @ Sigma @ w
    if variance <= 0:
        return 1e10
    return -excess_return / np.sqrt(variance)

# 3. Constraints:
#    sum(w) = 1: invest all risky-portfolio capital.
#    0 <= w_i <= 1: long-only; no short selling.
n = len(assets)
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, 1.0)] * n
initial_weights = np.ones(n) / n

# 4. Solve the constrained nonlinear optimization problem.
result = minimize(
    negative_sharpe,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-13, "maxiter": 5000, "disp": False},
)

if not result.success:
    raise RuntimeError(f"Optimization failed: {result.message}")

# 5. Clean numerical values close to zero, then renormalize.
weights = result.x.copy()
weights[np.abs(weights) < 1e-6] = 0.0
weights = weights / weights.sum()

# 6. Calculate portfolio statistics.
expected_excess_return = float(weights @ mu)
variance = float(weights @ Sigma @ weights)
risk = float(np.sqrt(variance))
sharpe_ratio = expected_excess_return / risk

# 7. Numerical validation.
assert np.isclose(weights.sum(), 1.0, atol=1e-8)
assert np.all(weights >= -1e-10)
assert np.all(np.isfinite(weights))
assert np.allclose(Sigma, Sigma.T, atol=1e-10)
assert np.linalg.eigvalsh(Sigma).min() > 0

# 8. Save outputs.
weights_df = pd.DataFrame({
    "Asset": assets,
    "Optimal_Weight": weights,
    "Optimal_Weight_pct": weights * 100,
})
weights_df.to_csv(WEIGHTS_FILE, index=False)

summary_df = pd.DataFrame([{
    "Expected_Excess_Return_pct_pa": expected_excess_return * 100,
    "Annual_Variance": variance,
    "Annual_Risk_StdDev_pct_pa": risk * 100,
    "Sharpe_Ratio": sharpe_ratio,
    "Weight_Sum": weights.sum(),
}])
summary_df.to_csv(SUMMARY_FILE, index=False)

print("\\nOPTIMAL RISKY PORTFOLIO")
print(weights_df.to_string(index=False, formatters={
    "Optimal_Weight": "{:.8f}".format,
    "Optimal_Weight_pct": "{:.4f}".format,
}))
print("\\nPORTFOLIO STATISTICS")
print(summary_df.to_string(index=False))
print("\\nAll numerical validation checks passed.")
