"""Task 2 - Variance decomposition using the Single Index Model.

Run:
    python task2_risk_decomposition.py DATA2026.xlsx

Outputs:
    task2_risk_decomposition.csv

Method:
    1. Calculate daily total returns from the Total Return Indices (TRI).
    2. Convert the Treasury bill APR (% p.a.) to a daily rate using 360 days.
    3. Calculate daily excess returns for the market and each stock.
    4. Estimate each stock's beta by OLS using daily excess returns.
    5. Annualize variances by multiplying daily variances by 250.
    6. Decompose total variance into systematic and idiosyncratic components.

Variances are reported in annual decimal-return-squared units. Annualized
volatility is also reported in percent per annum for easier interpretation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS_PER_YEAR = 250
DAYS_PER_YEAR_FOR_TBILL = 360

MARKET_COLUMN = "MSCI US"
RF_COLUMN = "US 3M TBILL (% p.a.)"

STOCKS = [
    "MMM", "GOOGL", "AXP", "AAPL", "CVX",
    "CSCO", "HON", "TRV", "V", "WMT",
]


def load_assignment_data(file_path: Path) -> pd.DataFrame:
    """Read and clean the DATA worksheet in DATA2026.xlsx."""
    if not file_path.exists():
        raise FileNotFoundError(f"Workbook not found: {file_path}")

    raw = pd.read_excel(file_path, sheet_name="DATA", header=None, engine="openpyxl")

    # Find the real header row rather than assuming a fixed row number.
    header_row = None
    for row_index in range(len(raw)):
        row_values = raw.iloc[row_index].astype(str).str.strip().tolist()
        if RF_COLUMN in row_values:
            header_row = row_index
            break

    if header_row is None:
        raise ValueError(f"Could not find '{RF_COLUMN}' in the DATA worksheet.")

    # Column B contains dates; columns C onward contain the rate and TRI series.
    headers = raw.iloc[header_row, 2:].tolist()
    data = raw.iloc[header_row + 1 :, 1 : 2 + len(headers)].copy()
    data.columns = ["Date"] + headers

    data = data.dropna(subset=["Date"]).copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    required_columns = [RF_COLUMN, MARKET_COLUMN, *STOCKS]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_counts = data[required_columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"Missing or non-numeric observations: {missing_counts.to_dict()}")

    return data


def calculate_daily_excess_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily TRI returns and subtract the daily Treasury bill rate."""
    assets = [MARKET_COLUMN, *STOCKS]
    returns = data[["Date"]].copy()

    for asset in assets:
        returns[asset] = data[asset].pct_change(fill_method=None)

    # Excel stores the bill rate in percentage points (e.g. 3.91 means 3.91%).
    returns["rf_daily"] = data[RF_COLUMN] / 100.0 / DAYS_PER_YEAR_FOR_TBILL

    # The first TRI observation has no preceding observation, so its return is missing.
    returns = returns.dropna(subset=assets + ["rf_daily"]).copy()

    returns["market_excess_daily"] = returns[MARKET_COLUMN] - returns["rf_daily"]
    for stock in STOCKS:
        returns[f"{stock}_excess_daily"] = returns[stock] - returns["rf_daily"]

    return returns.reset_index(drop=True)


def calculate_variance_decomposition(
    excess_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate annual total, systematic and idiosyncratic variances."""
    market_excess = excess_returns["market_excess_daily"].to_numpy(dtype=float)

    # Sample variance of the market's daily excess return, annualized by 250.
    market_variance_annual = (
        np.var(market_excess, ddof=1) * TRADING_DAYS_PER_YEAR
    )

    rows: list[dict[str, float | str | int]] = []

    # The market index is the factor itself: beta = 1 and residual variance = 0.
    rows.append(
        {
            "Asset": MARKET_COLUMN,
            "Beta": 1.0,
            "Total_variance_annual": market_variance_annual,
            "Systematic_variance_annual": market_variance_annual,
            "Idiosyncratic_variance_annual": 0.0,
            "Annual_volatility_pct": np.sqrt(market_variance_annual) * 100.0,
            "Systematic_share_pct": 100.0,
            "Idiosyncratic_share_pct": 0.0,
            "Decomposition_check": 0.0,
            "R_squared": 1.0,
            "Observations": len(market_excess),
        }
    )

    # Regress each stock's daily excess returns on daily market excess returns.
    X = sm.add_constant(market_excess)

    for stock in STOCKS:
        stock_excess = excess_returns[f"{stock}_excess_daily"].to_numpy(dtype=float)
        model = sm.OLS(stock_excess, X).fit()

        beta = float(model.params[1])
        residuals = np.asarray(model.resid, dtype=float)

        # IMPORTANT: variance annualization is daily variance * 250,
        # not variance of (daily return * 250).
        total_variance_annual = (
            np.var(stock_excess, ddof=1) * TRADING_DAYS_PER_YEAR
        )
        systematic_variance_annual = beta**2 * market_variance_annual
        idiosyncratic_variance_annual = (
            np.var(residuals, ddof=1) * TRADING_DAYS_PER_YEAR
        )

        decomposition_check = (
            total_variance_annual
            - systematic_variance_annual
            - idiosyncratic_variance_annual
        )

        rows.append(
            {
                "Asset": stock,
                "Beta": beta,
                "Total_variance_annual": total_variance_annual,
                "Systematic_variance_annual": systematic_variance_annual,
                "Idiosyncratic_variance_annual": idiosyncratic_variance_annual,
                "Annual_volatility_pct": np.sqrt(total_variance_annual) * 100.0,
                "Systematic_share_pct": (
                    systematic_variance_annual / total_variance_annual * 100.0
                ),
                "Idiosyncratic_share_pct": (
                    idiosyncratic_variance_annual / total_variance_annual * 100.0
                ),
                "Decomposition_check": decomposition_check,
                "R_squared": float(model.rsquared),
                "Observations": int(model.nobs),
            }
        )

    # The risk-free asset is treated as risk-free in the portfolio model.
    # Its excess return is zero by definition, so all three variances are zero.
    rows.append(
        {
            "Asset": "US 3M Treasury Bill (risk-free)",
            "Beta": 0.0,
            "Total_variance_annual": 0.0,
            "Systematic_variance_annual": 0.0,
            "Idiosyncratic_variance_annual": 0.0,
            "Annual_volatility_pct": 0.0,
            "Systematic_share_pct": np.nan,
            "Idiosyncratic_share_pct": np.nan,
            "Decomposition_check": 0.0,
            "R_squared": np.nan,
            "Observations": len(market_excess),
        }
    )

    results = pd.DataFrame(rows)

    # Fail loudly if the estimated variance decomposition does not reconcile.
    stock_checks = results.loc[results["Asset"].isin(STOCKS), "Decomposition_check"]
    if not np.allclose(stock_checks.to_numpy(), 0.0, atol=1e-10):
        raise ArithmeticError("Variance decomposition check failed for one or more stocks.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Task 2 variance decomposition under the Single Index Model."
    )
    parser.add_argument(
        "workbook",
        nargs="?",
        default="DATA2026.xlsx",
        help="Path to DATA2026.xlsx (default: DATA2026.xlsx).",
    )
    parser.add_argument(
        "--output",
        default="task2_risk_decomposition.csv",
        help="Output CSV path (default: task2_risk_decomposition.csv).",
    )
    args = parser.parse_args()

    data = load_assignment_data(Path(args.workbook))
    excess_returns = calculate_daily_excess_returns(data)
    results = calculate_variance_decomposition(excess_returns)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    display_columns = [
        "Asset",
        "Beta",
        "Total_variance_annual",
        "Systematic_variance_annual",
        "Idiosyncratic_variance_annual",
        "Annual_volatility_pct",
    ]
    print("\nTASK 2 — ANNUAL VARIANCE DECOMPOSITION")
    print(results[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))
    print(f"\nDaily return observations: {len(excess_returns)}")
    print(
        "Return sample: "
        f"{excess_returns['Date'].min().date()} to "
        f"{excess_returns['Date'].max().date()}"
    )
    print(f"Market annual variance: {results.loc[0, 'Total_variance_annual']:.10f}")
    print(
        "Maximum absolute decomposition difference: "
        f"{results.loc[results['Asset'].isin(STOCKS), 'Decomposition_check'].abs().max():.3e}"
    )
    print(f"\nSaved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
