
    market_variance = market_excess.var(ddof=1) * TRADING_DAYS
    beta_vector = np.array([betas[a] for a in ASSETS])

    covariance = market_variance * np.outer(beta_vector, beta_vector)
    for i, stock in enumerate(STOCKS, start=1):
        covariance[i, i] += residual_variances[stock]

    return covariance


def solve_min_variance(covariance, target_return=None, initial=None):
    """Solve a long-only, fully-invested minimum-variance problem."""
    n = len(ASSETS)
    if initial is None:
        initial = np.repeat(1.0 / n, n)

    constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
        "jac": lambda w: np.ones(n)
    }]

    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: w @ MU - target_return,
            "jac": lambda w: MU
        })

    result = minimize(
        fun=lambda w: w @ covariance @ w,
        x0=initial,
        jac=lambda w: 2.0 * covariance @ w,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 3000}
    )

    if not result.success:
        raise RuntimeError(result.message)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    covariance = build_covariance_matrix()

    covariance_df = pd.DataFrame(covariance, index=ASSETS, columns=ASSETS)
    covariance_df.index.name = "Asset"
    covariance_df.to_csv(OUTPUT_DIR / "task3_covariance_matrix.csv")

    # GMVP: minimize variance without fixing a target return.
    gmvp = solve_min_variance(covariance)
    w_gmvp = gmvp.x
    gmvp_return = w_gmvp @ MU
    gmvp_variance = w_gmvp @ covariance @ w_gmvp
    gmvp_risk = np.sqrt(gmvp_variance)

    pd.DataFrame({
        "Asset": ASSETS,
        "Weight": w_gmvp,
        "Weight_pct": w_gmvp * 100.0
    }).to_csv(OUTPUT_DIR / "task3_gmvp_weights.csv", index=False)

    # 21 target returns across the feasible long-only range.
    targets = np.linspace(MU.min(), MU.max(), 21)
    rows = []
    previous = w_gmvp.copy()

    for number, target in enumerate(targets, start=1):
        result = solve_min_variance(covariance, target, previous)
        weights = result.x
        port_return = weights @ MU
        port_variance = weights @ covariance @ weights
        port_risk = np.sqrt(port_variance)

        row = {
            "Portfolio": number,
            "Target_Excess_Return_pct": target * 100.0,
            "Actual_Excess_Return_pct": port_return * 100.0,
            "Variance": port_variance,
            "Standard_Deviation_pct": port_risk * 100.0
        }
        row.update({f"Weight_{a}": weights[i]
                    for i, a in enumerate(ASSETS)})
        rows.append(row)
        previous = weights

    frontier = pd.DataFrame(rows)
    frontier.to_csv(OUTPUT_DIR / "task3_frontier_21_portfolios.csv",
                    index=False)

    # Plot in expected excess return versus risk plane.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier["Standard_Deviation_pct"],
            frontier["Actual_Excess_Return_pct"],
            marker="o", label="Minimum Variance Frontier (21 portfolios)")
    ax.scatter(gmvp_risk * 100.0, gmvp_return * 100.0,
               marker="*", s=180, label="GMVP")
    ax.set_xlabel("Portfolio Risk (Annual Standard Deviation, %)")
    ax.set_ylabel("Expected Excess Return (% p.a.)")
    ax.set_title("Minimum Variance Frontier and GMVP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "task3_frontier.png", dpi=200)
    plt.close(fig)

    # Numerical validation.
    weights_columns = [f"Weight_{a}" for a in ASSETS]
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert np.allclose(frontier[weights_columns].sum(axis=1), 1.0, atol=1e-8)
    assert frontier[weights_columns].to_numpy().min() >= -1e-8
    assert np.allclose(frontier["Target_Excess_Return_pct"],
                       frontier["Actual_Excess_Return_pct"], atol=1e-8)

    print(f"GMVP expected excess return: {gmvp_return * 100:.4f}%")
    print(f"GMVP variance: {gmvp_variance:.8f}")
    print(f"GMVP standard deviation: {gmvp_risk * 100:.4f}%")
    print(f"Minimum covariance eigenvalue: {np.linalg.eigvalsh(covariance).min():.8f}")
    print("All optimization and constraint checks passed.")


if __name__ == "__main__":
    main()