# Task 4 — Optimal Risky Portfolio

## Objective
Find the long-only, fully-invested risky portfolio that maximizes the Sharpe ratio.

## Model
\[
\max_w \frac{w^T\mu}{\sqrt{w^T\Sigma w}}
\quad\text{s.t.}\quad
\mathbf{1}^T w=1,\; w_i\geq 0.
\]

- `mu`: annual expected excess returns from Task 1, in decimal units.
- `Sigma`: annual covariance matrix from the Single Index Model in Task 3.
- The 3-month US Treasury Bill is the risk-free asset and is not one of the 11 risky-asset weights.

## Files
- `task4_expected_excess_returns.csv`: annual expected excess returns.
- `task4_covariance_matrix.csv`: annual covariance matrix.
- `task4_optimal_risky_portfolio.py`: SciPy SLSQP optimization script.
- `task4_optimal_weights.csv`: optimal risky-asset composition.
- `task4_portfolio_summary.csv`: expected excess return, variance, risk and Sharpe ratio.

## Run
Install dependencies if necessary:
`pip install numpy pandas scipy`

Then run:
`python task4_optimal_risky_portfolio.py`

## Modeling note
The long-only and fully-invested constraints are carried forward from Task 3. The covariance matrix uses the Single Index Model assumption that individual residual returns are mutually uncorrelated and uncorrelated with the market.
