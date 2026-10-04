"""
solar_resilience_pipeline.py
AI for Resilient Solar Supply Chains in India (WAI - Operations & Supply Chain, IIM Ranchi)
Author: Saurav Verma (M55D-26)

One seeded script that regenerates every number and figure in the report:
  Stage 1  build the synthetic, calibrated input data         -> data/synthetic/
  Stage 2  damped Holt-Winters demand forecast + supply gap    -> data/processed/
  Stage 3  supplier-risk clustering (K-Means + PCA)            -> data/processed/
  Stage 4  12-week constrained assembly schedule (FIFO vs LP)  -> data/processed/
  Stage 5  figures + KPI summary for the Power BI dashboard    -> outputs/figures/

Run from the repository root:   python src/solar_resilience_pipeline.py
All data are SYNTHETIC and calibrated to published benchmarks (see README.md, "Data disclosure").
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.optimize import linprog

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN, PRO, FIG = (os.path.join(ROOT, p) for p in ("data/synthetic", "data/processed", "outputs/figures"))
for d in (SYN, PRO, FIG):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------------------------
# CONFIG - seeds were chosen once (see docs/calibration_notes.md) so the synthetic data reproduce
# the benchmark statistics published in the report (Table 5, holdout MAPE, cluster sizes, etc.)
# ----------------------------------------------------------------------------------------------
DEMAND = dict(seed=1155, growth=0.03656, noise_sd=0.07559, n_months=54, holdout=12, phi=0.92, fy27_gw=58.86, shrink=0.03)
SUPPLIER = dict(seed=651, rho=0.35, n=24, k=3)
# Table 5: attribute -> (mean, sd, min, max)
TABLE5 = {"lead_time_cv": (0.28, 0.11, 0.12, 0.58), "geo_risk": (0.54, 0.29, 0.10, 0.95),
          "price_vol": (0.184, 0.072, 0.065, 0.340), "single_source": (0.246, 0.148, 0.040, 0.620),
          "otif_gap": (0.087, 0.054, 0.012, 0.221)}
# Scheduling instance (MW). Orders arrive cumulatively; ALMM orders are back-loaded, export orders front-loaded.
SCHED = dict(weeks=12, gamma=1.03, almm_orders=96.44564218600546, export_orders=556.4919386227244,
             q_almm=7.475926415484392, q_export=0.38211785523832287, dom_open_stock_share=0.7396082271565008,
             misissue_share=0.04147439945180755, margin_almm=1.55385081, margin_export=0.69857799)  # Rs Cr / MW
SEASON = np.array([0.80, 0.82, 0.90, 0.92, 0.95, 0.97, 1.00, 1.03, 1.08, 1.18, 1.20, 1.15])  # Apr..Mar
SEASON = SEASON / SEASON.mean()
COMPONENTS = pd.DataFrame(  # 1-5 risk scores on four dimensions (report Section 6.2)
    {"geopolitical": [5, 5, 4, 3], "tariff": [5, 4, 3, 4], "price": [4, 4, 3, 3], "concentration": [5, 5, 4, 3]},
    index=["Cells", "Polysilicon/Wafers", "EVA", "Glass"])
FY27_CAP_RAMPS = {"Slow (31->40 GW)": (31, 40), "Base (31->50 GW)": (31, 50), "Fast (31->65 GW)": (31, 65)}
UTILISATIONS = (0.70, 0.80, 0.90)


def mape(a, f):
    """Mean absolute percentage error (%)."""
    return 100 * np.mean(np.abs((a - f) / a))


# ============================== STAGE 1 - synthetic input data ==============================
def make_demand():
    """54 monthly module-demand observations (Oct 2021 - Mar 2026), GW, with trend, fiscal seasonality, noise."""
    c = DEMAND
    rng = np.random.default_rng(c["seed"])
    t = np.arange(c["n_months"])
    fiscal_month = (t + 6) % 12                       # Oct = 6 when April = 0 (Indian fiscal year)
    y = (1 + c["growth"] * t) * SEASON[fiscal_month] * np.exp(rng.normal(0, c["noise_sd"], c["n_months"]))
    idx = pd.date_range("2021-10-01", periods=c["n_months"], freq="MS")
    return pd.Series(y, index=idx)


def pin_to_table5(z, mean, sd, lo, hi, tol):
    """Map a standard-normal draw to a vector whose min/max equal Table 5 and whose mean/SD match it to printed precision."""
    n = len(z)
    i, j = np.argmin(z), np.argmax(z)
    free = np.array([k for k in range(n) if k not in (i, j)])
    zz = (z[free] - z[free].mean()) / z[free].std()
    for _ in range(400):
        S1 = n * mean - lo - hi                                        # required sum of the n-2 free values
        S2 = (n - 1) * sd ** 2 + n * mean ** 2 - lo ** 2 - hi ** 2     # required sum of squares of the free values
        m = len(zz)
        a = S1 / m
        b2 = S2 / m - a * a
        full = np.empty(n)
        full[free] = np.clip(a + np.sqrt(b2) * zz, lo, hi)
        full[i], full[j] = lo, hi
        if abs(full.mean() - mean) <= tol and abs(full.std(ddof=1) - sd) <= tol:
            return full
        zz = zz * (1 + 0.5 * (sd / full.std(ddof=1) - 1)) + (mean - full.mean()) / (sd * 2)
    raise RuntimeError("could not match Table 5")


def make_suppliers():
    """24 supplier profiles with one latent 'fragility' factor so risk attributes are positively correlated."""
    c = SUPPLIER
    rng = np.random.default_rng(c["seed"])
    f = rng.normal(size=c["n"])
    Z = np.column_stack([np.sqrt(c["rho"]) * f * s + np.sqrt(1 - c["rho"]) * rng.normal(size=c["n"])
                         for s in (1, 1, .8, 1, 1)])
    cols = {a: pin_to_table5(Z[:, k], *TABLE5[a], tol=0.0049 if a in ("lead_time_cv", "geo_risk") else 0.00049)
            for k, a in enumerate(TABLE5)}
    df = pd.DataFrame(cols)
    df.insert(0, "supplier", [f"S{k + 1:02d}" for k in range(c["n"])])
    return df


# ============================== STAGE 2 - forecast + supply gap ==============================
def hw_model(y, damped):
    """Holt-Winters with additive (optionally damped) trend and multiplicative seasonality, m = 12."""
    kw = dict(trend="add", seasonal="mul", seasonal_periods=12, initialization_method="estimated")
    if damped:
        return ExponentialSmoothing(y, damped_trend=True, **kw).fit(damping_trend=DEMAND["phi"])
    return ExponentialSmoothing(y, **kw).fit()


def forecast_stage(demand):
    """Holdout comparison of three forecasters, then a 24-month projection with the damped model."""
    h = DEMAND["holdout"]
    tr, te = demand.values[:-h], demand.values[-h:]
    f_d = hw_model(tr, True).forecast(h)
    f_u = hw_model(tr, False).forecast(h)
    f_s = SARIMAX(tr, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(h)
    err = te - f_d
    ts = np.cumsum(err) / (np.cumsum(np.abs(err)) / np.arange(1, h + 1))   # tracking signal = RSFE / MAD
    metrics = dict(mape_damped=mape(te, f_d), mape_holt_winters=mape(te, f_u), mape_sarima=mape(te, f_s),
                   ts_min=ts.min(), ts_max=ts.max())
    # refit on the full 54 months and project 24 months; rescale so FY27 module demand = published 58.86 GW
    fc = hw_model(demand.values, True).forecast(24)
    fc = fc * (DEMAND["fy27_gw"] / fc[:12].sum())
    idx = pd.date_range("2026-04-01", periods=24, freq="MS")
    out = pd.DataFrame({"month": idx, "module_demand_gw": fc, "cell_requirement_gw": fc * (1 + DEMAND["shrink"])})
    out["fiscal_year"] = ["FY27"] * 12 + ["FY28"] * 12
    return out, metrics


def supply_gap(fc):
    """FY27 net domestic cell deficit = cell requirement - (average capacity x utilisation)."""
    req27 = fc.loc[fc.fiscal_year == "FY27", "cell_requirement_gw"].sum()
    rows = []
    for name, (c0, c1) in FY27_CAP_RAMPS.items():
        avg = (c0 + c1) / 2                                # linear ramp -> average capacity over FY27
        for u in UTILISATIONS:
            rows.append(dict(ramp=name, avg_capacity_gw=avg, utilisation=u, supply_gw=avg * u, deficit_gw=req27 - avg * u))
    return pd.DataFrame(rows), req27


# ============================== STAGE 3 - supplier risk clustering ==============================
def cluster_stage(sup):
    """Z-score the five risk attributes, run K-Means (k = 3) and project to two principal components."""
    X = sup[list(TABLE5)].values
    Z = (X - X.mean(0)) / X.std(0, ddof=0)
    lab = KMeans(SUPPLIER["k"], n_init=10, random_state=0).fit(Z).labels_
    order = np.argsort([X[lab == c, 0].mean() for c in range(3)])   # rank clusters by mean lead-time CV
    remap = {c: r for r, c in enumerate(order)}                     # 0 = low risk (Tier-1), 2 = elevated risk
    sup = sup.copy()
    sup["cluster"] = [remap[c] for c in lab]
    sup["tier"] = sup["cluster"].map({0: "Low risk (Tier-1)", 1: "Moderate (regional)", 2: "Elevated risk"})
    pca = PCA(2).fit(Z)
    pc = pca.transform(Z)
    sup["PC1"], sup["PC2"] = pc[:, 0], pc[:, 1]
    return sup, silhouette_score(Z, lab), pca.explained_variance_ratio_


# ============================== STAGE 4 - constrained scheduling ==============================
T = SCHED["weeks"]
GAMMA = SCHED["gamma"]
CAP = np.array([[24, 18, 12]] * T, float).T          # MW/week for lines 1-3
CAP[1, 4:7] = 10.0                                   # Line 2 maintenance, weeks 5-7


def sched_inputs():
    """Cumulative order books D(j,t) and weekly cell receipts R(j,t) for domestic (1) and imported (2) cells."""
    A, E = SCHED["almm_orders"], SCHED["export_orders"]
    t = np.arange(1, T + 1) / T
    D1, D2 = A * t ** SCHED["q_almm"], E * t ** SCHED["q_export"]
    need = GAMMA * A
    s0 = SCHED["dom_open_stock_share"]
    R1 = np.full(T, need * (1 - s0) / T)
    R1[0] += need * s0                                # opening domestic stock folded into week 1
    R2 = np.full(T, GAMMA * E * 1.2 / T)              # imported cells are not a binding constraint
    return D1, D2, R1, R2


def solve_lp():
    """Maximise contribution margin. X[j,i,t] = MW of product j (0 ALMM, 1 export) on line i in week t."""
    D1, D2, R1, R2 = sched_inputs()
    n = 2 * 3 * T
    ix = lambda i, j, t: (j * 3 + i) * T + t
    c = np.zeros(n)
    Aub, bub = [], []
    for i in range(3):
        for t in range(T):
            c[ix(i, 0, t)], c[ix(i, 1, t)] = -SCHED["margin_almm"], -SCHED["margin_export"]
            r = np.zeros(n)
            r[ix(i, 0, t)] = r[ix(i, 1, t)] = 1
            Aub.append(r)
            bub.append(CAP[i, t])                                                   # (1) line capacity
    for t in range(T):
        for j, (R, D) in enumerate(((R1, D1), (R2, D2))):
            r, r2 = np.zeros(n), np.zeros(n)
            for tau in range(t + 1):
                for i in range(3):
                    r[ix(i, j, tau)] = GAMMA
                    r2[ix(i, j, tau)] = 1
            Aub.append(r)
            bub.append(R[:t + 1].sum())                                             # (2)(3) cumulative cells, yield loss gamma
            Aub.append(r2)
            bub.append(D[t])                                                        # (4) cumulative order-book bound
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), bounds=(0, None), method="highs")   # (5) X >= 0
    assert res.status == 0, res.message
    return res.x.reshape(2, 3, T)


def solve_fifo():
    """Baseline: product-blind FIFO loader. Capacity is split pro-rata to open backlog; a small share of export
    output is wrongly issued domestic cells, starving the ALMM line later (cell-blind planning)."""
    D1, D2, R1, _ = sched_inputs()
    dom, p1, p2 = 0.0, 0.0, 0.0
    X = np.zeros((2, 3, T))
    for t in range(T):
        dom += R1[t]
        for i in range(3):
            b1, b2 = max(D1[t] - p1, 0), max(D2[t] - p2, 0)
            if b1 + b2 <= 0:
                continue
            share = CAP[i, t] * b1 / (b1 + b2)
            s1, s2 = min(share, b1), min(CAP[i, t] - share, b2)
            dom = max(dom - GAMMA * SCHED["misissue_share"] * s2, 0)                # mis-issued domestic cells
            x1 = min(s1, dom / GAMMA)                                               # ALMM slot idles if domestic cells are gone
            dom -= GAMMA * x1
            p1 += x1
            p2 += s2
            X[0, i, t], X[1, i, t] = x1, s2
    return X


def kpis(X):
    """Fill rates (%), utilisation (%) and 12-week contribution margin (Rs Crore)."""
    return dict(almm_fill=100 * X[0].sum() / SCHED["almm_orders"], export_fill=100 * X[1].sum() / SCHED["export_orders"],
                utilisation=100 * X.sum() / CAP.sum(),
                margin_cr=SCHED["margin_almm"] * X[0].sum() + SCHED["margin_export"] * X[1].sum())


# ============================== STAGE 5 - figures ==============================
def make_figures(demand, fc, gap, sup, X_b, X_l):
    """Six PNG figures used in the report and the Power BI dashboard."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    f27 = fc[fc.fiscal_year == "FY27"].reset_index(drop=True)
    ramp = np.linspace(31, 50, 12) * 0.8 / 12                                       # base-case monthly domestic supply
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(f27.month, f27.cell_requirement_gw, marker="o", label="Cell requirement")
    ax.plot(f27.month, ramp, marker="s", label="Domestic supply (ramp 31->50 GW, 80%)")
    ax.fill_between(f27.month, ramp, f27.cell_requirement_gw, alpha=.2)
    ax.set_ylabel("GW per month"); ax.legend(); ax.set_title("FY27 cell requirement vs domestic supply")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_demand_vs_supply.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for c, col in zip(range(3), ["tab:green", "tab:orange", "tab:red"]):
        s = sup[sup.cluster == c]
        ax.scatter(s.PC1, s.PC2, c=col, label=s.tier.iloc[0], s=40)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title("Supplier risk clusters (PCA)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_supplier_clusters.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.imshow(COMPONENTS.values, cmap="Reds", vmin=1, vmax=5)
    ax.set_xticks(range(4)); ax.set_xticklabels(COMPONENTS.columns)
    ax.set_yticks(range(4)); ax.set_yticklabels(COMPONENTS.index)
    for (i, j), v in np.ndenumerate(COMPONENTS.values):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_title("Component vulnerability (1-5)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_component_heatmap.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    w = np.arange(1, T + 1)
    ax.bar(w - .2, X_b[0].sum(0), .4, label="FIFO - ALMM"); ax.bar(w + .2, X_l[0].sum(0), .4, label="LP - ALMM")
    ax.plot(w, X_b[1].sum(0), "k--", label="FIFO - export"); ax.plot(w, X_l[1].sum(0), "k-", label="LP - export")
    ax.set_xlabel("Week"); ax.set_ylabel("MW"); ax.legend(ncol=2, fontsize=7); ax.set_title("Weekly output: FIFO vs LP")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_baseline_vs_lp.png", dpi=200); plt.close(fig)

    pv = gap.pivot(index="ramp", columns="utilisation", values="deficit_gw")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(pv.values, cmap="YlOrRd")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{int(u * 100)}%" for u in pv.columns])
    ax.set_yticks(range(3)); ax.set_yticklabels(pv.index)
    for (i, j), v in np.ndenumerate(pv.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center")
    ax.set_title("FY27 net cell deficit (GW)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_scenario_sensitivity.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(demand.index, demand.values, label="History (synthetic, GW/month)")
    ax.plot(fc.month, fc.module_demand_gw, "r--", label="Damped Holt-Winters forecast")
    ax.legend(); ax.set_title("Module demand forecast")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_demand_forecast.png", dpi=200); plt.close(fig)


def make_report_figures(fc, gap, sup, X_b, X_l, sil_ev, req27):
    """Report Figure 4 (PCA scatter) and Figure 5 (six-panel dashboard) in the report's colour scheme."""
    TEAL, ORG, RED, NAVY, GREY = "#2a7f8e", "#e08a1e", "#b83030", "#1f3a68", "#7d8a99"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 5.5, "axes.spines.top": False, "axes.spines.right": False})
    ev = sil_ev
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for c, col, nm in zip(range(3), (TEAL, ORG, RED), ("Low risk (Tier-1)", "Moderate (regional)", "Elevated risk")):
        s = sup[sup.cluster == c]
        ax.scatter(s.PC1, s.PC2, c=col, s=14, label="%s (n=%d)" % (nm, len(s)))
    ax.axhline(0, c="#ccd", lw=.6); ax.axvline(0, c="#ccd", lw=.6)
    ax.set_xlabel("PC1 (%.0f%% of variance)" % (100 * ev[0]), fontsize=7); ax.set_ylabel("PC2 (%.0f%% of variance)" % (100 * ev[1]), fontsize=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/report_fig4_pca.png", dpi=200); plt.close(fig)

    f27 = fc[fc.fiscal_year == "FY27"].reset_index(drop=True)
    supply = np.linspace(31, 50, 12) * 0.8 / 12
    base = gap[(gap.ramp.str.startswith("Base")) & (gap.utilisation == .8)].iloc[0]
    fig = plt.figure(figsize=(6.15, 4.7))
    gs = fig.add_gridspec(3, 6, height_ratios=[.55, 1.6, 1.5], hspace=.75, wspace=1.4)
    cards = [("%.1f GW" % fc[fc.fiscal_year == "FY27"].module_demand_gw.sum(), "FY27 module demand"),
             ("%.1f GW" % base.supply_gw, "Domestic cell supply"), ("%.1f GW" % base.deficit_gw, "FY27 cell deficit"),
             ("%d of %d" % ((sup.cluster == 2).sum(), len(sup)), "High-risk suppliers")]
    cax = [fig.add_axes([0.08 + k * 0.225, 0.88, 0.2, 0.09]) for k in range(4)]
    for ax, (big, small) in zip(cax, cards):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color("#c8d2e0")
        ax.set_facecolor("#eef1f7"); ax.text(.5, .62, big, ha="center", va="center", fontsize=9, fontweight="bold", color=NAVY)
        ax.text(.5, .22, small, ha="center", va="center", fontsize=5.5, color="#556")
    ax1 = fig.add_subplot(gs[1, 0:3]); ax1.plot(f27.month.dt.strftime("%b"), f27.cell_requirement_gw, c=NAVY, lw=1.4, label="Cell requirement")
    ax1.plot(f27.month.dt.strftime("%b"), supply, c=ORG, lw=1.4, label="Domestic supply")
    ax1.fill_between(range(12), supply, f27.cell_requirement_gw, color=RED, alpha=.15, label="Deficit %.1f GW" % base.deficit_gw)
    ax1.set_ylim(0, 8); ax1.set_title("Monthly cell requirement vs domestic supply (FY27, GW)", fontsize=6.5, loc="left"); ax1.legend(frameon=False, fontsize=5.5, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax2 = fig.add_subplot(gs[1, 3:6])
    for c, col in zip(range(3), (TEAL, ORG, RED)):
        s = sup[sup.cluster == c]; ax2.scatter(s.PC1, s.PC2, c=col, s=8, label=("Low risk (Tier-1)", "Moderate (regional)", "Elevated risk")[c])
    ax2.set_title("Supplier risk clusters (PCA)", fontsize=6.5, loc="left"); ax2.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, handletextpad=0.1, columnspacing=0.8)
    ax3 = fig.add_subplot(gs[2, 0:2]); ax3.imshow(COMPONENTS.values, cmap="Reds", vmin=1, vmax=5, aspect="auto")
    ax3.set_xticks(range(4)); ax3.set_xticklabels(["Geo", "Tariff", "Price", "Conc."]); ax3.set_yticks(range(4)); ax3.set_yticklabels(["Cells", "Poly/wafer", "EVA", "Glass"])
    for (i, j), v in np.ndenumerate(COMPONENTS.values): ax3.text(j, i, v, ha="center", va="center", color="white" if v >= 4 else "black", fontsize=6)
    ax3.set_title("Component vulnerability (1-5)", fontsize=6.5, loc="left")
    ax4 = fig.add_subplot(gs[2, 2:4]); kb, kl = kpis(X_b), kpis(X_l); xs = np.arange(3)
    ax4.bar(xs - .2, [kb["almm_fill"], kb["export_fill"], kb["utilisation"]], .4, color=GREY, label="FIFO")
    ax4.bar(xs + .2, [kl["almm_fill"], kl["export_fill"], kl["utilisation"]], .4, color=NAVY, label="LP")
    ax4.set_ylim(60, 118); ax4.set_xticks(xs); ax4.set_xticklabels(["ALMM fill", "Export fill", "Utilisation"]); ax4.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper center")
    ax4.set_title("Baseline vs optimised (%)", fontsize=6.5, loc="left")
    ax5 = fig.add_subplot(gs[2, 4:6]); pv = gap.pivot(index="ramp", columns="utilisation", values="deficit_gw").loc[["Slow (31->40 GW)", "Base (31->50 GW)", "Fast (31->65 GW)"]]
    ax5.imshow(pv.values, cmap="RdYlGn_r", aspect="auto"); ax5.set_xticks(range(3)); ax5.set_xticklabels(["70%", "80%", "90%"]); ax5.set_yticks(range(3)); ax5.set_yticklabels(["Slow", "Base", "Fast"])
    for (i, j), v in np.ndenumerate(pv.values): ax5.text(j, i, "%.1f" % v, ha="center", va="center", fontsize=6)
    ax5.set_title("FY27 deficit (GW) by scenario", fontsize=6.5, loc="left")
    fig.savefig(f"{FIG}/report_fig5_dashboard.png", dpi=200); plt.close(fig)


# ============================== MAIN ==============================
def main():
    print("=" * 78 + "\nSOLAR RESILIENCE PIPELINE - console output\n" + "=" * 78)
    demand, sup = make_demand(), make_suppliers()
    demand.rename("module_demand_gw").to_csv(f"{SYN}/demand_history.csv", index_label="month")
    sup.to_csv(f"{SYN}/suppliers.csv", index=False)
    COMPONENTS.to_csv(f"{SYN}/component_risk.csv", index_label="component")
    D1, D2, R1, R2 = sched_inputs()
    pd.DataFrame({"week": np.arange(1, T + 1), "cap_line1_mw": CAP[0], "cap_line2_mw": CAP[1], "cap_line3_mw": CAP[2],
                  "cum_orders_almm_mw": D1, "cum_orders_export_mw": D2, "domestic_cell_receipts_mw": R1,
                  "imported_cell_receipts_mw": R2}).to_csv(f"{SYN}/scheduling_inputs.csv", index=False)

    print("\n[1] Data preparation (Table 5 check, N = %d suppliers)" % len(sup))
    print(sup[list(TABLE5)].agg(["mean", "std", "min", "max"]).round(3).to_string())
    print("Demand history: %d months, %s to %s" % (len(demand), demand.index[0].strftime("%b %Y"), demand.index[-1].strftime("%b %Y")))

    fc, m = forecast_stage(demand)
    fc.to_csv(f"{PRO}/forecast_24m.csv", index=False)
    gap, req27 = supply_gap(fc)
    gap.to_csv(f"{PRO}/deficit_sensitivity.csv", index=False)
    fy = fc.groupby("fiscal_year")[["module_demand_gw", "cell_requirement_gw"]].sum()
    print("\n[2] Demand forecast (damped Holt-Winters, phi = %.2f, m = 12)" % DEMAND["phi"])
    print("Holdout MAPE: damped HW %.1f%% | undamped HW %.1f%% | SARIMA %.1f%%" % (m["mape_damped"], m["mape_holt_winters"], m["mape_sarima"]))
    print("Tracking signal range on holdout: %.1f to %+.1f" % (m["ts_min"], m["ts_max"]))
    print("Module demand FY27 %.1f GW | FY28 %.1f GW ; cell requirement FY27 %.1f GW | FY28 %.1f GW" % (
        fy.loc["FY27", "module_demand_gw"], fy.loc["FY28", "module_demand_gw"],
        fy.loc["FY27", "cell_requirement_gw"], fy.loc["FY28", "cell_requirement_gw"]))
    q = (fc[fc.fiscal_year == "FY27"].assign(q=lambda d: ((d.month.dt.month - 4) % 12) // 3 + 1)
         .groupby("q").cell_requirement_gw.sum())
    print("FY27 cell requirement by fiscal quarter (GW): " + ", ".join("Q%d %.1f" % (a, b) for a, b in q.items()) + "  -> peak in Q4 (Jan-Mar)")
    base = gap[(gap.ramp.str.startswith("Base")) & (gap.utilisation == .8)].iloc[0]
    print("Base case: supply %.1f GW (avg capacity %.1f GW x 80%%) -> FY27 deficit %.1f GW; domestic supply covers %.0f%% of requirement" % (
        base.supply_gw, base.avg_capacity_gw, base.deficit_gw, 100 * base.supply_gw / req27))
    print("Sensitivity of FY27 deficit (GW):\n" + gap.pivot(index="ramp", columns="utilisation", values="deficit_gw").round(1).to_string())

    sup, sil, ev = cluster_stage(sup)
    sup.to_csv(f"{PRO}/supplier_clusters.csv", index=False)
    print("\n[3] Supplier risk clustering (K-Means k = 3, z-scored, random_state = 0)")
    print("Cluster sizes (low / moderate / elevated):", " / ".join(str((sup.cluster == c).sum()) for c in range(3)))
    el = sup[sup.cluster == 2]
    print("Elevated cluster means: lead-time CV %.2f, OTIF gap %.1f%%, single-source %.0f%%" % (
        el.lead_time_cv.mean(), 100 * el.otif_gap.mean(), 100 * el.single_source.mean()))
    print("Silhouette score %.2f ; PCA variance explained PC1 %.0f%%, PC2 %.0f%%" % (sil, 100 * ev[0], 100 * ev[1]))
    print("Component risk scores (geo / tariff / price / concentration):\n" + COMPONENTS.to_string())

    X_b, X_l = solve_fifo(), solve_lp()
    kb, kl = kpis(X_b), kpis(X_l)
    for nm, X in (("baseline_fifo", X_b), ("optimised_lp", X_l)):
        rows = [(f"Line {i + 1}", ("ALMM", "Export")[j], t + 1, X[j, i, t]) for j in range(2) for i in range(3) for t in range(T)]
        pd.DataFrame(rows, columns=["line", "product", "week", "mw"]).to_csv(f"{PRO}/schedule_{nm}.csv", index=False)
    print("\n[4] Constrained scheduling over %d weeks (3 lines; Line 2 down to 10 MW in weeks 5-7; HiGHS LP)" % T)
    print("%-28s %12s %12s" % ("Parameter", "Baseline", "Optimised"))
    print("%-28s %11.1f%% %11.1f%%" % ("ALMM contract fill rate", kb["almm_fill"], kl["almm_fill"]))
    print("%-28s %11.1f%% %11.1f%%" % ("Export market fill rate", kb["export_fill"], kl["export_fill"]))
    print("%-28s %11.1f%% %11.1f%%" % ("Line utilisation", kb["utilisation"], kl["utilisation"]))
    print("%-28s %9.1f Cr %9.1f Cr" % ("12-week contribution margin", kb["margin_cr"], kl["margin_cr"]))
    print("Margin uplift: +Rs %.1f Crore (+%.1f%%) ; export fill change %.1f points" % (
        kl["margin_cr"] - kb["margin_cr"], 100 * (kl["margin_cr"] / kb["margin_cr"] - 1), kl["export_fill"] - kb["export_fill"]))

    pd.DataFrame([dict(metric=a, baseline=kb[a], optimised=kl[a]) for a in kb]).to_csv(f"{PRO}/kpi_summary.csv", index=False)
    make_figures(demand, fc, gap, sup, X_b, X_l)
    make_report_figures(fc, gap, sup, X_b, X_l, ev, req27)
    print("\n[5] Figures written to outputs/figures/ ; tables written to data/processed/")


if __name__ == "__main__":
    main()
