
# Empirical IO I, Problem Set 3 
# ------------------------------------------------------------------------------
# This script:
#   1) Generates synthetic data with correlated demand/supply shocks
#   2) Implements simulated shares, Jacobians, and two pricing solvers
#   3) Builds “observed” shares from simulated equilibrium
#   4) Estimates plain logit (OLS & 2SLS) and nested logit (2SLS)
#   5) Estimates a random-coefficients logit with pyblp (demand-only, joint, optimal IV)
#   6) Computes elasticities and diversion ratios (true vs estimated)
#   7) Simulates mergers (1–2 and 1–3), with and without efficiencies
#   8) Computes consumer surplus changes for a merger with cost savings
#
# Notes:
#   • Keep the simulation draws for heterogeneity FIXED across calls (crucial for convergence).
#   • Use log-sum-exp for numerical stability.
#   • The code is written to be clear; you can optimize vectorization if desired.
#
# Dependencies (install as needed):
#   pip install numpy pandas scipy statsmodels pyblp
#
# Author: La banda de IO
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import root
from scipy.special import logsumexp
import warnings

try:
    import pyblp
except Exception as e:
    pyblp = None
    warnings.warn(
        "pyblp could not be imported. Steps 8–14 require pyblp. "
        "Install with `pip install pyblp` and rerun."
    )
# ---- QUIET MODE ----
QUIET = True  # set to False if you want to see solver iterations again

import logging, warnings, contextlib, io, sys

# silence Python warnings and logging noise
if QUIET:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    for name in ["pyblp", "scipy", "urllib3", "numexpr"]:
        logging.getLogger(name).setLevel(logging.ERROR)

# helper to suppress stdout/stderr inside a with-block
@contextlib.contextmanager
def suppress_output():
    if not QUIET:
        yield
        return
    _stdout, _stderr = sys.stdout, sys.stderr
    try:
        fake = io.StringIO()
        with contextlib.redirect_stdout(fake), contextlib.redirect_stderr(fake):
            yield
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr

# -------------------
# 0. Reproducibility
# -------------------
rng = default_rng(1999)

# ---------------------
# 1. Generate the data
# ---------------------
T = 600                         # number of markets
J = 4                           # four inside goods per market
N = T * J

market_id = np.repeat(np.arange(1, T + 1), J)
product_id = np.tile(np.arange(1, J + 1), T)

# Technology indicators: goods 1–2 satellite, goods 3–4 wired
satellite = (product_id <= 2).astype(int)
wired = (product_id >= 3).astype(int)

# Exogenous product characteristic and cost shifter
x = np.abs(rng.standard_normal(N))
w = np.abs(rng.standard_normal(N))

# Demand and cost unobservables drawn jointly:
# (xi_{jt}, omega_{jt}) ~ N(0, Sigma) with Sigma = [[1, 0.25], [0.25, 1]]
Sigma = np.array([[1.0, 0.25], [0.25, 1.0]])
L = np.linalg.cholesky(Sigma)
Z = rng.standard_normal((2, N))
xi, omega = (L @ Z)
xi = xi.ravel()
omega = omega.ravel()

# Structural parameters
beta1 = 1.0     # coefficient on quality x
alpha = -2.0    # price coefficient
gamma0 = 0.5    # intercept in log marginal cost
gamma1 = 0.25   # slope on w in log marginal cost

# Random coefficients on satellite and on wired
beta_rc_mu = 4.0
beta_rc_sigma = 1.0

products = pd.DataFrame(
    {
        "market_id": market_id,
        "product_id": product_id,
        "firm_id": product_id,   # single-product firms baseline
        "satellite": satellite,
        "wired": wired,
        "x": x,
        "w": w,
        "xi": xi,
        "omega": omega,
    }
)

# Master draws for random coefficients (fixed across ALL calls)
M_MASTER = 5000
beta2_draws_master = rng.normal(loc=beta_rc_mu, scale=beta_rc_sigma, size=M_MASTER)  # sat coeff
beta3_draws_master = rng.normal(loc=beta_rc_mu, scale=beta_rc_sigma, size=M_MASTER)  # wired coeff


# ---------------------------------------------------
# 2(a). Simulated conditional shares & Jacobian d s/d p
# ---------------------------------------------------
def conditional_shares_matrix(p, x, sat, wired, xi, alpha, beta1, beta2_draws, beta3_draws):
    # Return a J x M matrix of conditional shares given prices p and random-coefficient draws.
    J = p.size
    M = beta2_draws.size
    V_base = beta1 * x + alpha * p + xi
    S = np.empty((J, M))

    # Stabilized denominator with outside good utility normalized to 0
    for m in range(M):
        V = V_base + beta2_draws[m] * sat + beta3_draws[m] * wired
        vmax = max(0.0, V.max())
        log_den = np.log1p(np.exp(V - vmax).sum()) + vmax  # log(1 + sum exp(V))
        S[:, m] = np.exp(V - log_den)
    return S


def jacobian_dsdp_market(p, x, sat, wired, xi, alpha, beta1, beta2_draws, beta3_draws):
    # Average Jacobian of shares with respect to prices (J x J) across simulated draws.
    S = conditional_shares_matrix(p, x, sat, wired, xi, alpha, beta1, beta2_draws, beta3_draws)
    J_, M = S.shape
    acc = np.zeros((J_, J_))
    for m in range(M):
        sm = S[:, m]
        acc += alpha * (np.diag(sm) - np.outer(sm, sm))
    return acc / M


def mean_shares_market(p, x, sat, wired, xi, alpha, beta1, beta2_draws, beta3_draws):
    return conditional_shares_matrix(p, x, sat, wired, xi, alpha, beta1, beta2_draws, beta3_draws).mean(axis=1)


# -------------------------------------------------------
# 2(c). Solve for equilibrium prices: two complementary ways
# -------------------------------------------------------
def _mc_from_df(df_t):
    # Log marginal cost: log c = gamma0 + gamma1 * w + omega / 8  ⇒  c = exp(log c).
    return np.exp(gamma0 + gamma1 * df_t["w"].to_numpy() + df_t["omega"].to_numpy() / 8.0)


def solve_prices_root_market(t, products, alpha, beta1, beta2_master, beta3_master, M=500, p0=None):
    # Root-finding on FOCs: p - mc - mu(p) = 0 where mu = -Δ^{-1} s.
    df_t = products.loc[products.market_id == t].copy()
    mc = _mc_from_df(df_t)
    J = len(df_t)
    if p0 is None:
        p0 = mc * 1.25
    d2 = beta2_master[:M]
    d3 = beta3_master[:M]

    x = df_t["x"].to_numpy()
    sat = df_t["satellite"].to_numpy()
    wir = df_t["wired"].to_numpy()
    xi = df_t["xi"].to_numpy()

    def F(p):
        s = mean_shares_market(p, x, sat, wir, xi, alpha, beta1, d2, d3)
        Delta = jacobian_dsdp_market(p, x, sat, wir, xi, alpha, beta1, d2, d3)
        mu = -np.linalg.solve(Delta, s)
        return p - mc - mu

    res = root(F, p0, method="hybr")
    return {"prices": res.x, "converged": res.success, "iterations": res.nfev, "method": "root"}


def solve_prices_ms_market(t, products, alpha, beta1, beta2_master, beta3_master, M=500, tol=1e-8, maxit=20000, p0=None):
    # Morrow–Skerlos ζ-iteration for single-product firms.
    df_t = products.loc[products.market_id == t].copy()
    mc = _mc_from_df(df_t)
    J = len(df_t)
    p = p0.copy() if p0 is not None else mc * 1.25
    zeta = p - mc

    d2 = beta2_master[:M]
    d3 = beta3_master[:M]

    x = df_t["x"].to_numpy()
    sat = df_t["satellite"].to_numpy()
    wir = df_t["wired"].to_numpy()
    xi = df_t["xi"].to_numpy()

    converged = False
    it = 0
    while it < maxit:
        it += 1

        S = conditional_shares_matrix(p, x, sat, wir, xi, alpha, beta1, d2, d3)
        s = S.mean(axis=1)
        s = np.clip(s, 1e-12, 1.0)  # protect against underflow
        Lambda = alpha * s
        Gamma = alpha * (S @ S.T) / M

        zeta_new = (Gamma @ zeta - s) / Lambda
        p_new = mc + zeta_new

        r = s + Lambda * zeta_new - Gamma @ zeta_new
        if np.max(np.abs(r)) < tol:
            converged = True
            zeta = zeta_new
            p = p_new
            break

        zeta = zeta_new
        p = p_new

    return {"prices": p, "converged": converged, "iterations": it, "method": "MS"}


def solve_all_markets(products, alpha, beta1, beta2_master, beta3_master, M=500, method="root"):
    out = []
    for t in np.unique(products.market_id):
        if method == "root":
            res = solve_prices_root_market(t, products, alpha, beta1, beta2_master, beta3_master, M=M)
        elif method == "ms":
            res = solve_prices_ms_market(t, products, alpha, beta1, beta2_master, beta3_master, M=M)
        else:
            raise ValueError("Unknown method: choose 'root' or 'ms'")
        out.append({"market": int(t), **res})
    return out


def write_observed_shares(products, results, alpha, beta1, beta2_master, beta3_master, M=500,
                          price_col="p_eq_ms", share_col="s_obs_ms"):
    products[price_col] = np.nan
    products[share_col] = np.nan
    d2 = beta2_master[:M]
    d3 = beta3_master[:M]
    for r in results:
        t = r["market"]
        idx = products.market_id == t
        df_t = products.loc[idx]
        p = r["prices"]
        s = mean_shares_market(
            p,
            df_t["x"].to_numpy(),
            df_t["satellite"].to_numpy(),
            df_t["wired"].to_numpy(),
            df_t["xi"].to_numpy(),
            alpha,
            beta1,
            d2,
            d3,
        )
        products.loc[idx, price_col] = p
        products.loc[idx, share_col] = s


# ----------------------------------------
# 3. Simulate equilibrium and “observed” shares
# ----------------------------------------
results_root = solve_all_markets(products, alpha, beta1, beta2_draws_master, beta3_draws_master, M=500, method="root")
results_ms = solve_all_markets(products, alpha, beta1, beta2_draws_master, beta3_draws_master, M=500, method="ms")

# (Optional) check that both methods agree closely
max_diff = np.max(
    np.concatenate(
        [np.abs(r["prices"] - m["prices"]) for r, m in zip(results_root, results_ms)]
    )
)
print(f"Max |Δp| across markets (root vs MS) = {max_diff:.3e}")

# Build “observed” shares from MS equilibrium
write_observed_shares(
    products, results_ms, alpha, beta1, beta2_draws_master, beta3_draws_master,
    M=500, price_col="p_eq_ms", share_col="s_obs_ms"
)

# ------------------------------
# 4–6. Plain and nested logit
# ------------------------------
# Berry outcome and within-nest logs from observed shares
def add_berry_and_within_logs(df):
    y = np.empty(N)
    ln_within_sat = np.zeros(N)
    ln_within_wir = np.zeros(N)
    for t in np.unique(df.market_id):
        idx = np.where(df.market_id.to_numpy() == t)[0]
        s = df.loc[idx, "s_obs_ms"].to_numpy()
        s0 = 1.0 - s.sum()
        s0 = max(s0, 1e-12)  # protect logs
        y[idx] = np.log(np.clip(s, 1e-15, 1.0)) - np.log(s0)

        satmask = df.loc[idx, "satellite"].to_numpy() == 1
        wirmask = df.loc[idx, "wired"].to_numpy() == 1
        S_sat = s[satmask].sum()
        S_wir = s[wirmask].sum()
        if S_sat > 0:
            ln_within_sat[idx[satmask]] = np.log(np.clip(s[satmask], 1e-15, 1.0)) - np.log(S_sat)
        if S_wir > 0:
            ln_within_wir[idx[wirmask]] = np.log(np.clip(s[wirmask], 1e-15, 1.0)) - np.log(S_wir)
    df = df.copy()
    df["y_berry"] = y
    df["ln_within_sat"] = ln_within_sat
    df["ln_within_wired"] = ln_within_wir
    return df

products = add_berry_and_within_logs(products)

# OLS: y = beta1 * x + alpha * p
y = products["y_berry"].to_numpy()
Xo = np.column_stack([products["x"].to_numpy(), products["p_eq_ms"].to_numpy()])
beta_ols, *_ = np.linalg.lstsq(Xo, y, rcond=None)
beta1_ols, alpha_ols = beta_ols

# 2SLS: instrument price with {x, w}
X = Xo.copy()
Z = np.column_stack([products["x"].to_numpy(), products["w"].to_numpy()])
PZ = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
beta_iv = np.linalg.inv(X.T @ PZ @ X) @ (X.T @ PZ @ y)
beta1_iv, alpha_iv = beta_iv

# Nested logit 2SLS: y = β*x + α*p + σ_sat*ln(s|sat) + σ_wir*ln(s|wir)
Xn = np.column_stack([
    products["x"].to_numpy(),
    products["p_eq_ms"].to_numpy(),
    products["ln_within_sat"].to_numpy(),
    products["ln_within_wired"].to_numpy(),
])
Zn = np.column_stack([
    products["x"].to_numpy(),
    products["w"].to_numpy(),            # instrument p only
    products["ln_within_sat"].to_numpy(),
    products["ln_within_wired"].to_numpy(),
])
PZn = Zn @ np.linalg.inv(Zn.T @ Zn) @ Zn.T
beta_nl = np.linalg.inv(Xn.T @ PZn @ Xn) @ (Xn.T @ PZn @ y)
beta1_nl, alpha_nl, sigma_sat, sigma_wir = beta_nl

print("Plain logit OLS:", dict(beta1=float(beta1_ols), alpha=float(alpha_ols)))
print("Plain logit 2SLS:", dict(beta1=float(beta1_iv), alpha=float(alpha_iv)))
print("Nested logit 2SLS:", dict(beta1=float(beta1_nl), alpha=float(alpha_nl),
                                   sigma_sat=float(sigma_sat), sigma_wir=float(sigma_wir)))

# ------------------------------------------------------
# 7. Own-price elasticities & diversion (true vs nested)
# ------------------------------------------------------

from exercise7_elasticities_diversion import run_exercise7

# Nested‑logit parameter estimates from Exercise 6:
params_nl = dict(
    alpha=alpha_nl,
    beta1=beta1_nl,
    sigma_sat=sigma_sat,
    sigma_wir=sigma_wir,
)

# "True" RC Jacobian (if you have jacobian_dsdp_market defined from your simulation code)
params_true = dict(
    alpha=alpha, beta1=beta1,
    d2=beta2_draws_master[:500], d3=beta3_draws_master[:500],
    xi_name="xi",
    jacobian_fn=jacobian_dsdp_market,
)

_ = run_exercise7(products, params_nl, params_true=params_true, h=1e-6, use_analytic_nl=True)


# --------------------------------------------------------
# 8. Random-coefficients logit with pyblp & instruments
# --------------------------------------------------------
if pyblp is not None:
    # Assemble product_data for pyblp
    product_data = pd.DataFrame(
        {
            "market_ids": products["market_id"].to_numpy(),
            "product_ids": products["product_id"].to_numpy(),
            "firm_ids": products["firm_id"].to_numpy(),   # single-product firms
            "prices": products["p_eq_ms"].to_numpy(),
            "shares": products["s_obs_ms"].to_numpy(),
            "x": products["x"].to_numpy(),
            "w": products["w"].to_numpy(),
            "satellite": products["satellite"].to_numpy(),
            "wired": products["wired"].to_numpy(),
        }
    )

    # Rival-sum BLP instruments and differentiation instruments
    blp_all = np.asarray(pyblp.build_blp_instruments(pyblp.Formulation("-1 + x + satellite + wired"), product_data))
    K = blp_all.shape[1] // 2
    blp_rival = blp_all[:, K:]

    diff_iv = np.asarray(pyblp.build_differentiation_instruments(pyblp.Formulation("-1 + x"), product_data))

    # Same-nest quality index: sum of x within the same technology (excluding self)
    same_nest_x = np.zeros(N)
    for t in np.unique(product_data["market_ids"]):
        idx = np.where(product_data["market_ids"] == t)[0]
        sat_mask = product_data.loc[idx, "satellite"].to_numpy() == 1
        wir_mask = product_data.loc[idx, "wired"].to_numpy() == 1
        x_sat = product_data.loc[idx[sat_mask], "x"].to_numpy()
        x_wir = product_data.loc[idx[wir_mask], "x"].to_numpy()
        same_nest_x[idx[sat_mask]] = (x_sat.sum() - x_sat)
        same_nest_x[idx[wir_mask]] = (x_wir.sum() - x_wir)

    xcol = products["x"].to_numpy().reshape(-1, 1)
    wcol = products["w"].to_numpy().reshape(-1, 1)
    satc = products["satellite"].to_numpy().reshape(-1, 1)
    wirc = products["wired"].to_numpy().reshape(-1, 1)
    sncol = same_nest_x.reshape(-1, 1)

    Zraw = np.hstack([blp_rival, diff_iv, sncol, xcol, wcol, satc, wirc])

    # Drop near-constant columns, then take a full column rank subset with QR
    stds = Zraw.std(axis=0)
    keep = np.where(stds > 1e-10)[0]
    Z1 = Zraw[:, keep]

    # Rank-revealing QR
    Q, R = np.linalg.qr(Z1, mode="reduced")
    diagR = np.abs(np.diag(R))
    r = np.where(diagR > 1e-10)[0].size
    if r > 0:
        idx_cols = np.arange(r)  # after QR with 'reduced', first r columns are independent
        Z = Z1[:, idx_cols]
    else:
        Z = np.zeros((Z1.shape[0], 0))

    # Attach instruments
    k_d = Z.shape[1]
    df_d = pd.DataFrame(Z, columns=[f"demand_instruments{j}" for j in range(k_d)])
    df_s = pd.DataFrame(xcol, columns=["supply_instruments0"])  # x excluded from costs except via w
    product_data = pd.concat([product_data, df_d, df_s], axis=1)

    # Model formulations:
    #  - Utility: intercept + prices + x + satellite  (omit wired to avoid perfect partition)
    #  - One random coefficient on satellite
    #  - Log marginal cost: intercept + w
    X1 = pyblp.Formulation("1 + prices + x + satellite")
    X2 = pyblp.Formulation("-1 + satellite")
    X3 = pyblp.Formulation("1 + w")

    integ = pyblp.Integration("halton", 500)
    sigma0 = np.array([[0.5]])

    if pyblp is not None:
        problem_d  = pyblp.Problem((X1, X2), product_data, integration=integ, add_exogenous=False)
        problem_js = pyblp.Problem((X1, X2, X3), product_data, integration=integ,
                               costs_type="log", add_exogenous=False)
    # ---- version-agnostic quiet optimizer ----
    # Some pyblp versions accept a second *positional* dict, some accept nothing.
        try:
            # Try passing a SciPy-style options dict positionally
            opt_quiet = pyblp.Optimization("l-bfgs-b", {"disp": False})
        except TypeError:
            # Fallback: no options argument supported
            opt_quiet = pyblp.Optimization("l-bfgs-b")

        # Demand-only
        with suppress_output():
            res_d = problem_d.solve(
                sigma=sigma0,
                method="2s",
                optimization=opt_quiet,
            )

        # Joint demand + supply
        with suppress_output():
            res_js = problem_js.solve(
                sigma=np.array(res_d.sigma),
                beta=np.array(res_d.beta),
                method="2s",
                optimization=opt_quiet,
            )

    # Feasible optimal IV
    with suppress_output():
        oi = res_js.compute_optimal_instruments(method="approximate")
        problem_opt = oi.to_problem()
        res_opt = problem_opt.solve(
            sigma=np.array(res_js.sigma),
            beta=np.array(res_js.beta),
            method="2s",
            optimization=opt_quiet,
        )

    # Helper to pull estimates and SEs
    def get_beta(res, name):
        labels = list(res.beta_labels)
        if name in labels:
            i = labels.index(name)
            return float(np.array(res.beta)[i]), float(np.array(res.beta_se)[i])
        return np.nan, np.nan

    def get_sigma(res):
        s = np.array(res.sigma)
        se = np.array(res.sigma_se)
        return float(s[0, 0]), float(se[0, 0])

    # Summaries
    bx_d, se_bx_d = get_beta(res_d, "x")
    ap_d, se_ap_d = get_beta(res_d, "prices")
    bsat_d, se_bsat_d = get_beta(res_d, "satellite")
    ssat_d, se_ssat_d = get_sigma(res_d)

    bx_js, se_bx_js = get_beta(res_js, "x")
    ap_js, se_ap_js = get_beta(res_js, "prices")
    bsat_js, se_bsat_js = get_beta(res_js, "satellite")
    ssat_js, se_ssat_js = get_sigma(res_js)

    bx_o, se_bx_o = get_beta(res_opt, "x")
    ap_o, se_ap_o = get_beta(res_opt, "prices")
    bsat_o, se_bsat_o = get_beta(res_opt, "satellite")
    ssat_o, se_ssat_o = get_sigma(res_opt)

    print("\npyBLP summaries (estimate (se))")
    def fmt(x): return f"{x:.3f}"
    print(f" Demand-only    : beta_x {fmt(bx_d)} ({fmt(se_bx_d)}), alpha {fmt(ap_d)} ({fmt(se_ap_d)}), "
          f"beta_sat {fmt(bsat_d)} ({fmt(se_bsat_d)}), sigma_sat {fmt(ssat_d)} ({fmt(se_ssat_d)})")
    print(f" Joint (supply) : beta_x {fmt(bx_js)} ({fmt(se_bx_js)}), alpha {fmt(ap_js)} ({fmt(se_ap_js)}), "
          f"beta_sat {fmt(bsat_js)} ({fmt(se_bsat_js)}), sigma_sat {fmt(ssat_js)} ({fmt(se_ssat_js)})")
    print(f" Opt. IV        : beta_x {fmt(bx_o)} ({fmt(se_bx_o)}), alpha {fmt(ap_o)} ({fmt(se_ap_o)}), "
          f"beta_sat {fmt(bsat_o)} ({fmt(se_bsat_o)}), sigma_sat {fmt(ssat_o)} ({fmt(se_ssat_o)})")

    # 9. Own-price elasticities & diversion from pyBLP estimates
    py_elasticities = np.asarray(res_js.compute_elasticities())
    own_est = np.zeros(J)
    D_est = np.zeros((J, J))

    for t in np.unique(product_data["market_ids"]):
        idx = np.where(product_data["market_ids"] == t)[0]
        s_t = product_data.loc[idx, "shares"].to_numpy()

        # pyBLP returns a JxJ matrix for each market flattened by market order:
        if py_elasticities.ndim == 2 and py_elasticities.shape[1] == J:
            E_t = py_elasticities[idx, :]     # J x J
            own_est += np.diag(E_t)
            for j in range(J):
                denom = -E_t[j, j] * s_t[j]
                if denom != 0:
                    D_est[:, j] += (E_t[:, j] * s_t) / denom
        else:
            # If only own elasticities returned
            own_est += py_elasticities[idx].ravel()

    m = len(np.unique(product_data["market_ids"]))
    own_est /= m
    if D_est.sum() != 0:
        D_est /= m
        # for j in range(J):
        #     D_est[j, j] = 0.0

    print("\\npyBLP average own-price elasticities (estimated):", np.round(own_est, 3))
    if D_est.sum() != 0:
        print("pyBLP average diversion ratios (estimated):\\n", np.round(D_est, 3))

    # 11. Merger simulations: 1–2 and 1–3 (no efficiencies)
    #def build_ownership_from_firm_ids(firm_ids, market_ids):
    #    # Build a block-diagonal JxJ ownership matrix from firm IDs by market.
    #    own = pyblp.build_ownership(firm_ids=firm_ids, market_ids=market_ids)
    #    return np.asarray(own)
    def build_ownership_from_firm_ids(firm_ids, market_ids):
        """
        Return ownership as an (N x J) stacked block matrix expected by your pyBLP version.
        For each market t, create a J_t x J_t block B with B[j,k] = 1 if products j and k
        in market t are owned by the same firm, else 0. Stack these blocks to make (N x J).
        """
        firm_ids = np.asarray(firm_ids)
        market_ids = np.asarray(market_ids)
        unique_markets = np.unique(market_ids)

        # infer J as max number of products per market (constant in this assignment: J=4)
        counts = np.array([np.sum(market_ids == t) for t in unique_markets])
        J = counts.max()

        N = firm_ids.size
        own = np.zeros((N, J), dtype=float)

        for t in unique_markets:
            idx = np.where(market_ids == t)[0]          # row indices for this market (length J_t)
            f = firm_ids[idx]                            # firm IDs for this market
            B = (f[:, None] == f[None, :]).astype(float) # J_t x J_t block of 0/1 ownership
            own[idx, :B.shape[1]] = B                    # write this block into the first J_t columns

        return own


    # Merge products a and b (map all b to a)
    def merged_firm_ids(pair, base_firm_ids):
        a, b = pair
        new_firm = base_firm_ids.copy()
        new_firm[product_data["product_ids"].to_numpy() == b] = a
        return new_firm

    own_12 = build_ownership_from_firm_ids(
        merged_firm_ids((1, 2), product_data["firm_ids"].to_numpy()),
        product_data["market_ids"].to_numpy()
    )
    own_13 = build_ownership_from_firm_ids(
        merged_firm_ids((1, 3), product_data["firm_ids"].to_numpy()),
        product_data["market_ids"].to_numpy()
    )

    p_pre_est = np.asarray(res_js.compute_prices()).ravel()
    p_post_12 = np.asarray(res_js.compute_prices(ownership=own_12)).ravel()
    p_post_13 = np.asarray(res_js.compute_prices(ownership=own_13)).ravel()

    def avg_by_product(v):
        return np.array([v[products["product_id"].to_numpy() == j].mean() for j in range(1, 5)])

    avg_p0_est = avg_by_product(p_pre_est)
    avg_p12 = avg_by_product(p_post_12)
    avg_p13 = avg_by_product(p_post_13)
    avg_dp12 = avg_by_product(p_post_12 - p_pre_est)
    avg_dp13 = avg_by_product(p_post_13 - p_pre_est)

    print("\\nAverage pre-merger prices by product:", np.round(avg_p0_est, 3))
    print("Average post-merger prices (1–2):", np.round(avg_p12, 3))
    print("Average post-merger prices (1–3):", np.round(avg_p13, 3))
    print("Average Δp (1–2):", np.round(avg_dp12, 3))
    print("Average Δp (1–3):", np.round(avg_dp13, 3))
    print("Average Δp difference (1–2 minus 1–3):", np.round(avg_dp12 - avg_dp13, 3))

    # 13–14. Merger 1–2 with 15% marginal cost reduction for firms 1 & 2
    # Use costs from the joint model, not the ex ante mc function
    costs = np.asarray(res_js.compute_costs()).ravel()
    merger_costs = costs.copy()
    mask12 = np.isin(products["product_id"].to_numpy(), [1, 2])
    merger_costs[mask12] *= 0.85
    p_post_12_eff = np.asarray(res_js.compute_prices(ownership=own_12, costs=merger_costs)).ravel()
    avg_dp12_eff = avg_by_product(p_post_12_eff - p_pre_est)
    print("\\nAverage Δp for merger 1–2 with 15% cost reduction:", np.round(avg_dp12_eff, 3))

    # Consumer surplus change using mean utilities from the joint model (δ_hat) and the estimated σ on satellite
    delta_hat = np.asarray(res_js.delta).ravel()
    beta_labels = list(res_js.beta_labels)
    beta_vec = np.asarray(res_js.beta).ravel()

    # Map coefficients by label
    def get_coeff(name):
        return float(beta_vec[beta_labels.index(name)])

    beta0_hat = get_coeff("1")
    alpha_hat = get_coeff("prices")
    betax_hat = get_coeff("x")
    betasat_hat = get_coeff("satellite")
    sigma_hat = float(np.asarray(res_js.sigma)[0, 0])

    # Hold xi fixed; adjust delta for counterfactual prices
    xi_hat = delta_hat - (
        beta0_hat + betax_hat * products["x"].to_numpy() + betasat_hat * products["satellite"].to_numpy()
        + alpha_hat * p_pre_est
    )
    delta_cf = (
        beta0_hat + betax_hat * products["x"].to_numpy() + betasat_hat * products["satellite"].to_numpy()
        + alpha_hat * p_post_12_eff + xi_hat
    )

    def cs_from_delta(delta, sat, alpha_price, sigma, M=1000, seed=1999):
        rng_local = default_rng(seed)
        draws = rng_local.normal(loc=0.0, scale=sigma, size=M)
        acc = 0.0
        for m in range(M):
            V = delta + draws[m] * sat
            # log(1 + sum exp(V)) with outside good normalized to 0
            acc += logsumexp(np.concatenate(([0.0], V)))
        return (acc / M) / (-alpha_price)

    # Average CS change per market
    Delta_CS = 0.0
    for t in np.unique(products["market_id"]):
        idx = np.where(products["market_id"].to_numpy() == t)[0]
        sat_t = products.loc[idx, "satellite"].to_numpy()
        Delta_CS += cs_from_delta(delta_cf[idx], sat_t, alpha_hat, sigma_hat) - cs_from_delta(delta_hat[idx], sat_t, alpha_hat, sigma_hat)
    Delta_CS_avg = Delta_CS / len(np.unique(products["market_id"]))
    print("Average change in consumer surplus (per consumer) for merger 1–2 with 15% cost reduction:",
          f"{Delta_CS_avg:.3f}")
else:
    print("\\npyBLP not available: skipping steps 8–14 (estimation and merger simulations).")
