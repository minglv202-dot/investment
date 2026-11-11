# Task 2 — Single Index Model: Variance Decomposition

This project completes Task 2 of the KU Leuven Investments Assignment 2026.
It estimates the annual total, systematic, and idiosyncratic variances of the
MSCI US Index and the ten researched US stocks using the Single Index Model.

## Model

For stock *i* on day *t*:

\[
R_{i,t}-R_{f,t}=a_i+\beta_i(R_{m,t}-R_{f,t})+\epsilon_{i,t}.
\]

The variance decomposition is:

\[
\operatorname{Var}(R_i-R_f)
=\beta_i^2\operatorname{Var}(R_m-R_f)
+\operatorname{Var}(\epsilon_i).
\]

Annualization follows the assignment's 250-trading-day convention:

\[
\sigma^2_{annual}=250\,s^2_{daily}.
\]

Thus:

- Total variance: `250 * sample_variance(stock_daily_excess_return)`.
- Systematic variance: `beta**2 * annual_market_variance`.
- Idiosyncratic variance: `250 * sample_variance(OLS_daily_residuals)`.

Sample variances use `ddof=1`. With an OLS regression including an intercept,
the residuals are orthogonal to the market factor in-sample. Using the same
sample-variance convention for returns and residuals makes the sample variance
decomposition reconcile (up to floating-point precision).

Variance columns are expressed in **annual decimal-return-squared units**.
For intuition, the script also reports annualized volatility in percent:
`sqrt(annual_variance) * 100`.

The market index is the factor itself, so its beta is 1 and its idiosyncratic
variance is 0. The 3-month Treasury bill is shown as a risk-free asset with zero
variance in the portfolio model; the historical bill-rate series is used to
construct daily excess returns, not treated as a risky asset-return index.

## Data and conventions

- Input: `DATA2026.xlsx`, worksheet `DATA`.
- Stock and market returns are calculated from Total Return Indices.
- Treasury bill APR is supplied in percentage points (for example, `3.91`
  means 3.91% p.a.) and converted to a daily rate as `APR / 100 / 360`.
- Daily excess returns are regressed using OLS with an intercept.
- Daily variances are annualized by multiplying by 250, as stated in the assignment.
- The script checks that total variance equals systematic plus idiosyncratic
  variance for each stock.

## Run

Install dependencies:

```bash
pip install numpy pandas openpyxl statsmodels
```

Place `DATA2026.xlsx` next to the script, then run:

```bash
python task2_risk_decomposition.py DATA2026.xlsx
```

Optional output path:

```bash
python task2_risk_decomposition.py DATA2026.xlsx --output outputs/task2_risk_decomposition.csv
```

Do not publish the course-provided workbook to a public repository unless you
have permission to share it. The script and README can be committed separately.
