## Task 4 — Optimal Risky Portfolio

The optimal risky portfolio was obtained by maximizing the Sharpe ratio across the eleven risky assets. The annual expected excess return vector estimated in Task 1 and the annual covariance matrix constructed using the Single Index Model in Task 3 were used as inputs. The portfolio was assumed to be fully invested in risky assets and short selling was not allowed.

The optimization problem is

\[
\max_{w}\; \frac{w^{\top}\mu}{\sqrt{w^{\top}\Sigma w}}
\quad\text{subject to}\quad
\mathbf{1}^{\top}w=1,\qquad w_i\geq 0.
\]

The resulting optimal portfolio has an expected excess return of **5.3785% per annum**, an annual standard deviation of **15.8585%**, and a Sharpe ratio of **0.3392**. Its annual variance is **0.0251492**.

The portfolio allocates approximately 46.22% to the MSCI US Index, 9.52% to MMM, 11.22% to AAPL, 17.37% to HON, 5.27% to TRV, and 10.40% to WMT. The optimal weights of GOOGL, AXP, CVX, CSCO and V are zero under the imposed long-only constraints.

These results identify the portfolio with the highest estimated risk-adjusted excess return among the feasible risky-asset portfolios, conditional on the expected returns, covariance estimates and constraints used. The result should therefore be interpreted in light of the Single Index Model assumptions and estimation uncertainty.
