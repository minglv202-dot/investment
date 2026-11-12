"""
Task 5 – Closed-form optimal risky portfolio (long-only).

Inputs:
  inputs/task1_results.csv
  inputs/task3_covariance_matrix.csv

Method:
  1. Compute the unconstrained tangency portfolio in closed form.
  2. Solve the long-only problem by enumerating candidate active sets.
  3. Verify the KKT conditions for the selected active set.
  4. Export portfolio weights, summary statistics and KKT diagnostics.

All expected returns and covariance entries are annualized and expressed
in decimal units (e.g. 5.55% = 0.0555).
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = [
    "MSCI US", "MMM", "GOOGL", "AXP", "AAPL", "CVX",
    "CSCO", "HON", "TRV", "V", "WMT"
]
TOL = 1e-9


def load_inputs(input_dir: Path):
    """Load expected excess returns and the annual covariance matrix."""
    returns_df = pd.read_csv(input_dir / "task1_results.csv")
    covariance_df = pd.read_csv(input_dir / "task3_covariance_matrix.csv", index_col=0)

    returns_df = returns_df.set_index("Asset")
    mu = returns_df.loc[ASSETS, "Expected_excess_return_pct"].astype(float).to_numpy() / 100.0
    sigma = covariance_df.loc[ASSETS, ASSETS].astype(float).to_numpy()

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("Covariance matrix is not symmetric.")
    if np.min(np.linalg.eigvalsh(sigma)) <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    return mu, sigma


def portfolio_metrics(weights, mu, sigma):
    """Return annual expected excess return, variance, volatility and Sharpe."""
    expected_excess = float(weights @ mu)
    variance = float(weights @ sigma @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = expected_excess / volatility if volatility > 0 else np.nan
    return expected_excess, variance, volatility, sharpe


def unconstrained_closed_form(mu, sigma):
    """
    Fully invested unconstrained tangency portfolio:
        w = Sigma^{-1} mu / (1' Sigma^{-1} mu).

    np.linalg.solve is used instead of explicitly forming Sigma^{-1}.
    """
    q = np.linalg.solve(sigma, mu)
    denominator = float(np.sum(q))
    if abs(denominator) < TOL:
        raise ValueError("Cannot normalize the unconstrained portfolio.")
    return q / denominator


def solve_long_only_by_active_sets(mu, sigma):
    """
    Solve:
        min_x x' Sigma x
        subject to mu' x = 1 and x >= 0.

    For each candidate active set A, the equality-constrained solution is:
        x_A = Sigma_A^{-1} mu_A /
              (mu_A' Sigma_A^{-1} mu_A).

    The KKT conditions are checked for every candidate. The resulting x is
    then rescaled to fully invested weights w = x / sum(x).
    """
    n = len(mu)
    candidates = []

    for size in range(1, n + 1):
        for active in combinations(range(n), size):
            active = np.array(active, dtype=int)
            inactive = np.array([i for i in range(n) if i not in active], dtype=int)

            sigma_a = sigma[np.ix_(active, active)]
            mu_a = mu[active]

            try:
                q_a = np.linalg.solve(sigma_a, mu_a)
            except np.linalg.LinAlgError:
                continue

            denominator = float(mu_a @ q_a)
            if denominator <= TOL:
                continue

            x = np.zeros(n)
            x[active] = q_a / denominator

            # The selected active variables must be strictly positive.
            if np.any(x[active] <= TOL):
                continue
            if abs(float(mu @ x) - 1.0) > 1e-7:
                continue

            # KKT multipliers: 2 Sigma x - lambda mu - nu = 0,
            # nu >= 0, nu_i*x_i = 0.
            lam = 2.0 / denominator
            nu = 2.0 * (sigma @ x) - lam * mu
            if inactive.size and np.any(nu[inactive] < -1e-7):
                continue
            if np.max(np.abs(nu[active])) > 1e-7:
                continue

            # Convert the normalized-return vector x to sum-to-one weights.
            x_sum = float(np.sum(x))
            if x_sum <= 0:
                continue
            weights = x / x_sum

            candidates.append({
                "active": active,
                "inactive": inactive,
                "x": x,
                "weights": weights,
                "lambda": lam,
                "nu": nu,
                "denominator": denominator,
            })

    if not candidates:
        raise RuntimeError("No long-only active set satisfied the KKT conditions.")

    # Strictly convex objective implies a unique solution. If numerical
    # boundary degeneracy yields more than one candidate, retain the one
    # with the smallest objective value.
    best = min(candidates, key=lambda item: float(item["x"] @ sigma @ item["x"]))
    return best


def main():
    root = Path(__file__).resolve().parent
    input_dir = root / "inputs"
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    mu, sigma = load_inputs(input_dir)

    # Step 1: unconstrained closed-form result (diagnostic only).
    w_unconstrained = unconstrained_closed_form(mu, sigma)
    unconstrained_metrics = portfolio_metrics(w_unconstrained, mu, sigma)

    # Step 2: long-only closed-form active-set result.
    solution = solve_long_only_by_active_sets(mu, sigma)
    w = solution["weights"]
    expected_excess, variance, volatility, sharpe = portfolio_metrics(w, mu, sigma)

    # Step 3: export weights.
    weights_df = pd.DataFrame({
        "Asset": ASSETS,
        "Weight": w,
        "Weight_pct": 100.0 * w,
    })
    weights_df.to_csv(output_dir / "task5_optimal_weights.csv", index=False)

    unconstrained_df = pd.DataFrame({
        "Asset": ASSETS,
        "Weight": w_unconstrained,
        "Weight_pct": 100.0 * w_unconstrained,
    })
    unconstrained_df.to_csv(output_dir / "task5_unconstrained_weights.csv", index=False)

    # Step 4: KKT diagnostics for the selected active set.
    active_set = set(solution["active"].tolist())
    kkt_rows = []
    for i, asset in enumerate(ASSETS):
        is_active = i in active_set
        kkt_rows.append({
            "Asset": asset,
            "Status": "Active (positive weight)" if is_active else "Inactive (zero weight)",
            "Weight_pct": 100.0 * w[i],
            "Normalized_return_variable_x": solution["x"][i],
            "KKT_multiplier_nu": solution["nu"][i],
            "Complementary_slackness_nu_times_x": solution["nu"][i] * solution["x"][i],
        })
    kkt_df = pd.DataFrame(kkt_rows)
    kkt_df.to_csv(output_dir / "task5_kkt_validation.csv", index=False)

    # Step 5: portfolio summary and Task 4 comparison.
    task4_reference = {
        "Sharpe_ratio": 0.33915572,
        "Expected_excess_return": 0.0537850167,
        "Variance": 0.0251492083,
        "Risk_volatility": 0.1585850191,
    }
    summary_rows = [
        ("Method", "Closed-form active-set solution"),
        ("Sharpe_ratio", sharpe),
        ("Expected_excess_return", expected_excess),
        ("Variance", variance),
        ("Risk_volatility", volatility),
        ("Sum_of_weights", float(np.sum(w))),
        ("Active_assets", ", ".join(ASSETS[i] for i in solution["active"])),
        ("Inactive_assets", ", ".join(ASSETS[i] for i in solution["inactive"])),
        ("Task4_Sharpe_ratio_reference", task4_reference["Sharpe_ratio"]),
        ("Task4_expected_excess_return_reference", task4_reference["Expected_excess_return"]),
        ("Task4_variance_reference", task4_reference["Variance"]),
        ("Task4_risk_volatility_reference", task4_reference["Risk_volatility"]),
        ("Unconstrained_Sharpe_ratio_diagnostic", unconstrained_metrics[3]),
        ("Unconstrained_expected_excess_return_diagnostic", unconstrained_metrics[0]),
        ("Unconstrained_risk_volatility_diagnostic", unconstrained_metrics[2]),
        ("KKT_lambda", solution["lambda"]),
        ("Minimum_eigenvalue_covariance", float(np.min(np.linalg.eigvalsh(sigma)))),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    summary_df.to_csv(output_dir / "task5_portfolio_summary.csv", index=False)

    # Human-readable run report.
    print("TASK 5 – CLOSED-FORM LONG-ONLY OPTIMAL RISKY PORTFOLIO")
    print("=" * 62)
    print(f"Sharpe ratio:                  {sharpe:.8f}")
    print(f"Expected excess return:        {expected_excess:.8%}")
    print(f"Annual variance:               {variance:.10f}")
    print(f"Annual volatility (risk):      {volatility:.8%}")
    print(f"Sum of portfolio weights:     {np.sum(w):.10f}")
    print("\nOptimal portfolio composition:")
    for asset, weight in zip(ASSETS, w):
        print(f"  {asset:10s} {weight:10.6%}")

    print("\nUnconstrained diagnostic:")
    print(f"Sharpe ratio:                  {unconstrained_metrics[3]:.8f}")
    print("Negative weights:", [
        ASSETS[i] for i, value in enumerate(w_unconstrained) if value < -TOL
    ])

    print("\nKKT check:")
    print("Active set:", ", ".join(ASSETS[i] for i in solution["active"]))
    print("Inactive set:", ", ".join(ASSETS[i] for i in solution["inactive"]))
    print("Minimum inactive multiplier:",
          min(solution["nu"][solution["inactive"]], default=0.0))
    print("Maximum active multiplier absolute value:",
          max((abs(solution["nu"][i]) for i in solution["active"]), default=0.0))

    print(f"\nCSV files saved in: {output_dir}")


if __name__ == "__main__":
    main()
