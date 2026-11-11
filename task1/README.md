# Investments Assignment 2026 — Task 1

Estimate the expected annual excess returns of the MSCI US Index and the 10 researched US stocks using the **Single Index Model**.

## Model

For stock \(i\), the Single Index Model is

\[
R_{i,t}-R_{f,t}=a_i+\beta_i(R_{m,t}-R_{f,t})+\epsilon_{i,t}.
\]

The beta is estimated by OLS using historical daily excess returns. The analysts' annual alpha forecasts are then combined with the expected annual market risk premium:

\[
E(R_i)-R_f=\alpha_i+\beta_i\big(E(R_m)-R_f\big).
\]

Assignment inputs:

- Expected annual MSCI US return: **9.75%**
- Annual risk-free rate: **4.20%**
- Expected market risk premium: **5.55%**
- Stock-return annualization convention: **250 trading days per year**
- Treasury bill APR conversion: **360 days per year**
- Visa (V) alpha forecast: **0.00%**, as confirmed for this calculation

## Data

Put the assignment workbook `DATA2026.xlsx` in this folder. The script reads the `DATA` worksheet and extracts the TRI series and Treasury bill rate.

The workbook contains the MSCI US index and 10 stocks: MMM, GOOGL, AXP, AAPL, CVX, CSCO, HON, TRV, V, and WMT.

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the calculation:

```bash
python task1_expected_excess_returns.py DATA2026.xlsx
```

The script writes `task1_results.csv` with full-precision results. The terminal output is displayed to four decimal places.

## Method

1. Calculate daily TRI returns: \(R_{i,t}=TRI_{i,t}/TRI_{i,t-1}-1\).
2. Convert the Treasury bill APR from percent per annum into a daily decimal rate: \(r_{f,t}=(APR_t/100)/360\).
3. Calculate daily excess returns for each stock and MSCI US.
4. Estimate each stock's beta by OLS regression of stock excess return on MSCI US excess return.
5. Calculate expected annual excess returns using the analyst alpha forecasts and the 5.55% market risk premium.

The Treasury bill rate recorded on date \(t\) is matched to the TRI return from date \(t-1\) to date \(t\). This is the date-alignment convention used by the script.

## Important distinction

The historical regression intercept is not substituted for the analysts' alpha forecast. The regression is used to estimate **beta**; the assignment's analyst alpha forecasts are used in the expected-return calculation.
