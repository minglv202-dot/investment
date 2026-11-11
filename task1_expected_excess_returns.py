"""Task 1 - Expected Excess Returns using the Single Index Model.

Run:
    python task1_expected_excess_returns.py DATA2026.xlsx

The script:
1. Reads the DATA worksheet from the assignment workbook.
2. Calculates daily total returns from the Total Return Indices (TRI).
3. Converts the annual Treasury bill APR (% p.a.) to a daily rate using 360 days.
4. Calculates daily stock and market excess returns.
5. Estimates each stock's beta by OLS regression against MSCI US excess returns.
6. Combines beta estimates with the analysts' annual alpha forecasts to calculate
   expected annual excess returns.
7. Saves a CSV table to task1_results.csv.

All intermediate calculations retain full floating-point precision. Rounding is
applied only for display; the CSV stores the unrounded numerical results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# -----------------------------------------------------------------------------
# Assignment assumptions and analyst forecasts
# -----------------------------------------------------------------------------
EXPECTED_MARKET_RETURN_PCT = 9.75       # % per annum
RISK_FREE_RATE_PCT = 4.20              # % per annum
TRADING_DAYS_PER_YEAR = 250
DAYS_PER_YEAR_FOR_TBILL = 360

# Alpha forecasts from Assignment2026.pdf, in percentage points per annum.
# Visa (V) is set to 0.00% based on the user's confirmation.
ALPHA_FORECASTS_PCT = {
    "MMM": 1.25,
    "GOOGL": -0.50,
    "AXP": -1.00,
    "AAPL": 0.90,
    "CVX": 0.00,
    "CSCO": -0.80,
    "HON": 1.25,
    "TRV": 0.50,
    "V": 0.00,
    "WMT": 1.00,
}

MARKET_COLUMN = "MSCI US"
RF_COLUMN = "US 3M TBILL (% p.a.)"


def load_assignment_data(file_path: Path) -> pd.DataFrame:
    """Load and clean the assignment workbook's DATA sheet.

    The supplied workbook has title rows above the actual column names. The
    function locates the row containing 'US 3M TBILL (% p.a.)' instead of
    relying on a fixed Excel header argument.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find input workbook: {file_path}\n"
            "Place DATA2026.xlsx in this folder or pass its path as an argument."
        )

    raw = pd.read_excel(file_path, sheet_name="DATA", header=None, engine="openpyxl")

    header_row = None
    for row_index in range(len(raw)):
        row_values = raw.iloc[row_index].astype(str).str.strip().tolist()
        if RF_COLUMN in row_values:
            header_row = row_index
            break

    if header_row is None:
        raise ValueError(f"Could not find the column heading '{RF_COLUMN}' in the DATA sheet.")

    # In this workbook, column B contains dates, while columns C:N contain
    # the risk-free rate and the market/stock Total Return Indices.
    headers = raw.iloc[header_row, 2:].tolist()
    data = raw.iloc[header_row + 1 :, 1 : 2 + len(headers)].copy()
    data.columns = ["Date"] + headers

    data = data.dropna(subset=["Date"]).copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    numeric_columns = [RF_COLUMN, *headers[1:]]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing = data[[RF_COLUMN, MARKET_COLUMN, *ALPHA_FORECASTS_PCT.keys()]].isna().sum()
    if missing.any():
        missing_columns = missing[missing > 0].to_dict()
        raise ValueError(f"Missing/non-numeric observations found: {missing_columns}")

    missing_assets = set(ALPHA_FORECASTS_PCT) - set(data.columns)
    if missing_assets:
        raise ValueError(f"Assets with alpha forecasts but no TRI column: {sorted(missing_assets)}")

    return data


def calculate_daily_excess_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily asset returns and daily excess returns.

    The Treasury bill rate recorded on date t is matched to the TRI return
    from t-1 to t. The rate is provided in percent per annum, so it is divided
    by 100 and then by 360 as instructed in the assignment.
    """
    asset_columns = [MARKET_COLUMN, *ALPHA_FORECASTS_PCT.keys()]

    returns = data[["Date"]].copy()
    for asset in asset_columns:
        returns[asset] = data[asset].pct_change(fill_method=None)

    returns["rf_daily"] = data[RF_COLUMN] / 100.0 / DAYS_PER_YEAR_FOR_TBILL

    # The first observation has no previous TRI and therefore no daily return.
    returns = returns.dropna(subset=asset_columns + ["rf_daily"]).copy()

    returns["market_excess_daily"] = returns[MARKET_COLUMN] - returns["rf_daily"]
    for stock in ALPHA_FORECASTS_PCT:
        returns[f"{stock}_excess_daily"] = returns[stock] - returns["rf_daily"]

    return returns.reset_index(drop=True)


def estimate_betas(excess_returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate stock betas using OLS: stock excess return ~ market excess return.

    Daily excess returns are multiplied by 250 before regression so that the
    regression intercept is expressed on an annualized simple-return basis.
    This common scaling does not change the estimated beta or R-squared.
    """
    x_annualized = excess_returns["market_excess_daily"] * TRADING_DAYS_PER_YEAR
    x_design = sm.add_constant(x_annualized.rename("market_excess_annualized"))

    rows: list[dict[str, float | str | int]] = []
    for stock in ALPHA_FORECASTS_PCT:
        y_annualized = excess_returns[f"{stock}_excess_daily"] * TRADING_DAYS_PER_YEAR
        model = sm.OLS(y_annualized, x_design).fit()

        rows.append(
            {
                "Asset": stock,
                "Beta": float(model.params["market_excess_annualized"]),
                "Historical_intercept_annualized_pct": float(model.params["const"] * 100),
                "R_squared": float(model.rsquared),
                "Observations": int(model.nobs),
            }
        )

    return pd.DataFrame(rows)


def calculate_expected_excess_returns(beta_table: pd.DataFrame) -> pd.DataFrame:
    """Combine analyst alpha forecasts and estimated betas with market premium."""
    market_risk_premium_pct = EXPECTED_MARKET_RETURN_PCT - RISK_FREE_RATE_PCT

    results = beta_table.copy()
    results["Analyst_alpha_pct"] = results["Asset"].map(ALPHA_FORECASTS_PCT)
    results["Market_risk_premium_pct"] = market_risk_premium_pct

    # All values here are in percentage points (% p.a.), not decimal returns.
    results["Expected_excess_return_pct"] = (
        results["Analyst_alpha_pct"]
        + results["Beta"] * results["Market_risk_premium_pct"]
    )
    results["Expected_total_return_pct"] = (
        results["Expected_excess_return_pct"] + RISK_FREE_RATE_PCT
    )

    # Put the rows in the same order as the assignment's alpha table/data.
    results = results[
        [
            "Asset",
            "Analyst_alpha_pct",
            "Beta",
            "Market_risk_premium_pct",
            "Expected_excess_return_pct",
            "Expected_total_return_pct",
            "Historical_intercept_annualized_pct",
            "R_squared",
            "Observations",
        ]
    ]

    # Add the market index as a reference asset: beta=1 by definition relative
    # to itself, and its expected excess return is the market risk premium.
    market_row = pd.DataFrame(
        [
            {
                "Asset": MARKET_COLUMN,
                "Analyst_alpha_pct": np.nan,
                "Beta": 1.0,
                "Market_risk_premium_pct": market_risk_premium_pct,
                "Expected_excess_return_pct": market_risk_premium_pct,
                "Expected_total_return_pct": EXPECTED_MARKET_RETURN_PCT,
                "Historical_intercept_annualized_pct": np.nan,
                "R_squared": 1.0,
                "Observations": int(beta_table["Observations"].iloc[0]),
            }
        ]
    )
    results = pd.concat([market_row, results], ignore_index=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Task 1 expected excess returns using the Single Index Model."
    )
    parser.add_argument(
        "workbook",
        nargs="?",
        default="DATA2026.xlsx",
        help="Path to DATA2026.xlsx (default: DATA2026.xlsx in the current directory).",
    )
    parser.add_argument(
        "--output",
        default="task1_results.csv",
        help="Output CSV path (default: task1_results.csv).",
    )
    args = parser.parse_args()

    workbook_path = Path(args.workbook)
    output_path = Path(args.output)

    data = load_assignment_data(workbook_path)
    excess_returns = calculate_daily_excess_returns(data)
    beta_table = estimate_betas(excess_returns)
    results = calculate_expected_excess_returns(beta_table)

    # Keep full precision in the CSV; only the console display is rounded.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, float_format="%.12f")

    display_columns = [
        "Asset",
        "Analyst_alpha_pct",
        "Beta",
        "Expected_excess_return_pct",
        "Expected_total_return_pct",
    ]
    print("\nTASK 1 — EXPECTED EXCESS RETURNS (% p.a.)")
    print(results[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nReturn observations used per stock: {len(excess_returns)}")
    print(f"Sample period: {excess_returns['Date'].min().date()} to {excess_returns['Date'].max().date()}")
    print(f"Market risk premium: {EXPECTED_MARKET_RETURN_PCT - RISK_FREE_RATE_PCT:.4f}% p.a.")
    print(f"\nSaved full-precision results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
