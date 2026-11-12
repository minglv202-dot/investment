# Task 5 – Closed-form optimal risky portfolio

This folder contains a reproducible Python implementation of Task 5.

## Folder structure

```text
task5_closed_form_solution/
├── task5_closed_form_solution.py
├── inputs/
│   ├── task1_results.csv
│   └── task3_covariance_matrix.csv
└── outputs/
    ├── task5_optimal_weights.csv
    ├── task5_unconstrained_weights.csv
    ├── task5_portfolio_summary.csv
    └── task5_kkt_validation.csv
```

## Requirements

- Python 3.9+
- numpy
- pandas

Install dependencies:

```bash
pip install numpy pandas
```

## Run

From this folder:

```bash
python task5_closed_form_solution.py
```

## Method

Let `mu` be the vector of expected annual excess returns and `Sigma` the
annual covariance matrix.

The unconstrained fully-invested tangency portfolio is

\[
w_{unc}=\frac{\Sigma^{-1}\mu}{\mathbf{1}'\Sigma^{-1}\mu}.
\]

This solution may contain negative weights, so it does not necessarily
satisfy the assignment's no-short-selling condition.

For the long-only solution, the program uses the scale-invariant
transformation

\[
\min_x x'\Sigma x,\qquad \text{s.t. }\mu'x=1,\;x\geq 0.
\]

It enumerates candidate active sets. For a candidate active set `A`, the
closed-form equality-constrained solution is

\[
x_A=\frac{\Sigma_A^{-1}\mu_A}
{\mu_A'\Sigma_A^{-1}\mu_A},
\qquad x_{A^c}=0.
\]

The KKT conditions are checked, including non-negativity of the inactive
multipliers and complementary slackness. The normalized-return vector is
then converted to fully-invested portfolio weights by

\[
w=\frac{x}{\mathbf{1}'x}.
\]

The covariance matrix is positive definite, so the objective is strictly
convex and a KKT-feasible solution is the global optimum.

## Important interpretation

The unconstrained result is reported as a diagnostic only. The final Task 5
answer is the long-only active-set portfolio, which is directly comparable
to the no-short-selling numerical maximization in Task 4.

All return and risk outputs are annualized. Expected returns and covariance
inputs are converted to decimal units before calculation.
