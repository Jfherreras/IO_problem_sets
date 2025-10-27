# exercise7_elasticities_diversion.py
# --------------------------------------------------------------
# Exercise 7: Own-price elasticities & diversion ratios
#   - "True" model (via provided Jacobian callback)
#   - Estimated Nested Logit (analytic Jacobian)
# --------------------------------------------------------------

import numpy as np

# Map your column names here if they differ
COLS = {
    "market": "market_id",
    "price": "p_eq_ms",
    "x": "x",
    "sat": "satellite",
    "wir": "wired",
    "share_obs": "s_obs_ms",
    "xi": "xi",  # only needed if your "true" Jacobian needs xi
}

# ---------- Nested Logit shares (two nests: satellite, wired) ----------
def nested_logit_shares(p, x, sat, wir, alpha, beta, sigma_sat, sigma_wir):
    """
    Return:
      s      : product shares (J,)
      s_cond : dict of within-nest shares ('sat','wir'), (J,) arrays with zeros outside the nest
      s_nest : dict of nest shares {'sat': scalar, 'wir': scalar}
    """
    v = beta * x + alpha * p
    J = len(p)
    s = np.zeros(J)

    idx_sat = (sat == 1)
    idx_wir = (wir == 1)

    s_cond = {"sat": np.zeros(J), "wir": np.zeros(J)}
    s_nest = {"sat": 0.0, "wir": 0.0}

    iv_sat = 0.0
    if np.any(idx_sat):
        den_sat = np.exp(v[idx_sat] / (1.0 - sigma_sat)).sum()
        s_cond_sat = np.zeros(J)
        s_cond_sat[idx_sat] = np.exp(v[idx_sat] / (1.0 - sigma_sat)) / den_sat
        s_cond["sat"] = s_cond_sat
        iv_sat = den_sat ** (1.0 - sigma_sat)

    iv_wir = 0.0
    if np.any(idx_wir):
        den_wir = np.exp(v[idx_wir] / (1.0 - sigma_wir)).sum()
        s_cond_wir = np.zeros(J)
        s_cond_wir[idx_wir] = np.exp(v[idx_wir] / (1.0 - sigma_wir)) / den_wir
        s_cond["wir"] = s_cond_wir
        iv_wir = den_wir ** (1.0 - sigma_wir)

    denom = 1.0 + iv_sat + iv_wir  # outside inclusive value is 1
    if iv_sat > 0:
        s_nest["sat"] = iv_sat / denom
        s[idx_sat] = s_cond["sat"][idx_sat] * s_nest["sat"]
    if iv_wir > 0:
        s_nest["wir"] = iv_wir / denom
        s[idx_wir] = s_cond["wir"][idx_wir] * s_nest["wir"]

    return s, s_cond, s_nest

# ---------- Analytic Nested-Logit Jacobian d s / d p (J x J) ----------
def nested_logit_jacobian(p, x, sat, wir, alpha, beta, sigma_sat, sigma_wir):
    """
    Two-nest NL derivative:
        ∂s_j/∂p_m = alpha * s_j * (1{j=m} - s_m) / (1 - sigma_{nest(m)})
    """
    s, _, _ = nested_logit_shares(p, x, sat, wir, alpha, beta, sigma_sat, sigma_wir)
    J = len(p)
    Jhat = np.zeros((J, J))

    denom_sat = max(1e-12, 1.0 - sigma_sat)
    denom_wir = max(1e-12, 1.0 - sigma_wir)

    for m in range(J):
        denom = denom_sat if sat[m] == 1 else denom_wir
        for j in range(J):
            Jhat[j, m] = alpha * s[j] * ((1.0 if j == m else 0.0) - s[m]) / denom

    return Jhat, s

# ---------- Diversion & elasticities across markets ----------
def compute_diversion_and_elasticities(products,
                                       params_nl,
                                       params_true=None,
                                       h=1e-6,
                                       use_analytic_nl=True):
    """
    products: pandas DataFrame with columns specified in COLS
    params_nl: dict(alpha, beta1, sigma_sat, sigma_wir)
    params_true (optional): dict(alpha, beta1, d2, d3, xi_name='xi', jacobian_fn=callable)

    Returns:
      dict with: own_true, own_hat, D_true, D_hat, outside_true, outside_hat
    """
    c = COLS
    markets = np.array(sorted(products[c["market"]].unique()))
    # Check constant J across markets:
    J_counts = products.groupby(c["market"]).size().values
    if not np.all(J_counts == J_counts[0]):
        raise ValueError(f"Exercise 7 expects constant #products per market; found {np.unique(J_counts)}")
    J = int(J_counts[0])
    m = len(markets)

    own_true = np.zeros(J)
    own_hat  = np.zeros(J)
    D_true   = np.zeros((J, J))
    D_hat    = np.zeros((J, J))

    have_true = params_true is not None and callable(params_true.get("jacobian_fn", None))

    # NL params
    alpha_nl = params_nl["alpha"]
    beta1_nl = params_nl["beta1"]
    sigma_sat = params_nl["sigma_sat"]
    sigma_wir = params_nl["sigma_wir"]

    for t in markets:
        idx = products[c["market"]].to_numpy() == t
        p   = products.loc[idx, c["price"]].to_numpy().astype(float)
        x_t = products.loc[idx, c["x"]].to_numpy().astype(float)
        sat_t = products.loc[idx, c["sat"]].to_numpy().astype(int)
        wir_t = products.loc[idx, c["wir"]].to_numpy().astype(int)

        # ----- TRUE model (if provided) -----
        if have_true:
            xi_name = params_true.get("xi_name", c["xi"])
            xi_t = products.loc[idx, xi_name].to_numpy().astype(float)
            alpha = params_true["alpha"]
            beta1 = params_true["beta1"]
            d2 = params_true["d2"]
            d3 = params_true["d3"]
            Delta = params_true["jacobian_fn"](p, x_t, sat_t, wir_t, xi_t, alpha, beta1, d2, d3)

            s_true = products.loc[idx, c["share_obs"]].to_numpy().astype(float)
            s_true = np.clip(s_true, 1e-12, 1.0)
            own_true += np.diag(Delta) * (p / s_true)

            for j in range(J):
                denom = -Delta[j, j]
                if abs(denom) > 0:
                    D_true[:, j] += Delta[:, j] / denom

        # ----- Estimated Nested Logit -----
        if use_analytic_nl:
            Jhat, s_hat = nested_logit_jacobian(p, x_t, sat_t, wir_t,
                                                alpha_nl, beta1_nl, sigma_sat, sigma_wir)
        else:
            # finite-difference fallback
            s_hat, _, _ = nested_logit_shares(p, x_t, sat_t, wir_t,
                                              alpha_nl, beta1_nl, sigma_sat, sigma_wir)
            Jhat = np.zeros((J, J))
            for j in range(J):
                pb = p.copy()
                step = h * max(1.0, abs(p[j]))
                pb[j] += step
                s2, _, _ = nested_logit_shares(pb, x_t, sat_t, wir_t,
                                               alpha_nl, beta1_nl, sigma_sat, sigma_wir)
                Jhat[:, j] = (s2 - s_hat) / step

        s_hat = np.clip(s_hat, 1e-12, 1.0)
        own_hat += np.diag(Jhat) * (p / s_hat)
        for j in range(J):
            denom = -Jhat[j, j]
            if abs(denom) > 0:
                D_hat[:, j] += Jhat[:, j] / denom

    # Average across markets
    if have_true:
        own_true /= m
        D_true  /= m
        np.fill_diagonal(D_true, 0.0)  # presentation

    own_hat /= m
    D_hat  /= m
    np.fill_diagonal(D_hat, 0.0)

    # Diversion to outside (by column j)
    outside_true = (1.0 - D_true.sum(axis=0)) if have_true else None
    outside_hat  = 1.0 - D_hat.sum(axis=0)

    return dict(
        own_true=own_true if have_true else None,
        own_hat=own_hat,
        D_true=D_true if have_true else None,
        D_hat=D_hat,
        outside_true=outside_true,
        outside_hat=outside_hat,
    )

# ---------- Pretty print helpers ----------
def _pp_matrix(M, name):
    if M is None:
        print(f"{name}: [not computed]")
        return
    print(f"{name}:")
    print(np.round(M, 3))

def _pp_vector(v, name):
    if v is None:
        print(f"{name}: [not computed]")
        return
    print(f"{name}: {np.round(v, 3)}")

# ---------- Public runner ----------
def run_exercise7(products, params_nl, params_true=None, h=1e-6, use_analytic_nl=True):
    res = compute_diversion_and_elasticities(products, params_nl, params_true, h, use_analytic_nl)

    _pp_vector(res["own_true"], "Avg own-price elasticities (true)")
    _pp_vector(res["own_hat"],  "Avg own-price elasticities (nested est.)")

    _pp_matrix(res["D_true"], "Avg diversion ratios (true)")
    _pp_matrix(res["D_hat"],  "Avg diversion ratios (nested est.)")

    _pp_vector(res["outside_true"], "Avg diversion to outside (true, by j)")
    _pp_vector(res["outside_hat"],  "Avg diversion to outside (nested est., by j)")

    return res
