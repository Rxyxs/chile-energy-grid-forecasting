"""Figuras de los datos reales: validación de la reconstrucción y precio por barra.

    python -m src.data.report_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import cen_daily_reports as C  # noqa: E402
from src.data import real_sources as R  # noqa: E402
from src.data.real_panel import BAR_NODES, PRICE_TARGET, load_bar_cmg, load_system_hourly  # noqa: E402

OUT_DIR = R.PROJECT_ROOT / "data" / "processed"
BAR_COLORS = ["#d62728", "#8c8c3a", "#6e548d", "#2a78d6", "#eb6834", "#1baf7a"]


def _full_months(ts: pd.Series) -> tuple[pd.Series, list[str]]:
    month = ts.dt.strftime("%Y-%m")
    days = ts.groupby(month).apply(lambda t: t.dt.date.nunique())
    return days, [m for m, d in days.items() if d >= 0.95 * pd.Period(m).days_in_month]


def reconstruction_check() -> pd.DataFrame:
    """Meses completos: energía reconstruida / energía oficial (datos.gob.cl) y
    costo marginal medio de Crucero reconstruido vs. el de los reportes de la CNE."""
    s = load_system_hourly()
    days, full = _full_months(s["timestamp"])
    month = s["timestamp"].dt.strftime("%Y-%m")
    scale = pd.Series({m: pd.Period(m).days_in_month / days[m] for m in full})
    sums = s.groupby(month)[["solar_generation_mwh", "wind_generation_mwh", "demand_mwh"]].sum().loc[full].mul(scale, axis=0)
    g = R.load_generation().set_index("month").reindex(full)
    out = pd.DataFrame({
        "solar": sums["solar_generation_mwh"] / g["solar_mwh"],
        "wind": sums["wind_generation_mwh"] / g["wind_mwh"],
        "total": sums["demand_mwh"] / g["total_mwh"],
    })
    cmg = load_bar_cmg()
    cru = cmg[cmg["node"] == "Crucero_220kV"]
    cdays, cfull = _full_months(cru["timestamp"])
    cmean = cru.groupby(cru["timestamp"].dt.strftime("%Y-%m"))[PRICE_TARGET].mean().loc[cfull]
    cne = R.load_cmg().set_index("month")
    out = out.join(pd.DataFrame({"crucero_reconstructed": cmean,
                                 "crucero_cne": cne["cmg_usd_mwh"].reindex(cmean.index),
                                 "cne_source": cne["source"].reindex(cmean.index)}), how="outer")
    return out.sort_index()


def plot_reconstruction_check(check: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    idx = pd.PeriodIndex(check.index, freq="M").to_timestamp()
    for col, color, label in (("solar", "#eda100", "Solar"), ("wind", "#1baf7a", "Wind"), ("total", "#4a3aa7", "Total")):
        ax.plot(idx, (check[col] - 1) * 100, marker="o", ms=3, color=color, label=label, linewidth=1.4)
    ax.axhline(0, color="#898781", linewidth=1)
    ax.set_ylabel("Reconstructed − official (%)")
    ax.set_title("Monthly energy from the PDF charts vs. official SEN generation")
    ax.legend()
    ax = axes[1]
    c = check.dropna(subset=["crucero_reconstructed", "crucero_cne"])
    table = c["cne_source"] == "tabla"
    ax.scatter(c.loc[~table, "crucero_cne"], c.loc[~table, "crucero_reconstructed"], color="#9aa0a6", s=22,
               label="CNE value read from text/OCR/derived")
    ax.scatter(c.loc[table, "crucero_cne"], c.loc[table, "crucero_reconstructed"], color="#d62728", s=26,
               label="CNE value read from a table")
    lim = [0, max(c["crucero_cne"].max(), c["crucero_reconstructed"].max()) * 1.05]
    ax.plot(lim, lim, color="#898781", linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("CNE monthly report (USD/MWh)")
    ax.set_ylabel("Mean of reconstructed hours (USD/MWh)")
    ax.set_title("Crucero 220 kV monthly marginal cost")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_reconstruction_check.png", dpi=130)
    plt.close(fig)


def plot_price_profile_by_bar() -> None:
    cmg = load_bar_cmg()
    cmg["hour"] = cmg["timestamp"].dt.hour
    cmg["year"] = cmg["timestamp"].dt.year
    years = sorted(cmg["year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(4 * len(years), 4.2), sharey=True)
    for ax, year in zip(axes, years):
        prof = cmg[cmg["year"] == year].pivot_table(index="hour", columns="node", values=PRICE_TARGET, aggfunc="mean")
        for color, node in zip(BAR_COLORS, BAR_NODES.values()):
            if node in prof:
                ax.plot(prof.index, prof[node], color=color, linewidth=1.8,
                        label=node.replace("_220kV", "").replace("_", " "))
        ax.set_title(str(year))
        ax.set_xlabel("Hour of day")
        ax.set_xticks([0, 6, 12, 18, 23])
    axes[0].set_ylabel("Mean marginal cost (USD/MWh)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Real hourly marginal cost by bar, north (red) to south (green)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_price_profile_by_bar.png", dpi=130)
    plt.close(fig)


def extraction_quality_summary() -> pd.Series:
    q = pd.read_csv(C.REPORT_QUALITY_PATH)
    return pd.Series({
        "days_with_report": len(q),
        "days_with_6plus_bars": int((q["cmg_bars"] >= 6).sum()),
        "cmg_resid_median_usd": q["cmg_max_resid_usd"].median(),
        "cmg_resid_p99_usd": q["cmg_max_resid_usd"].quantile(0.99),
        "gen_scale_err_median": q["gen_max_rel_err"].median(),
        "gen_scale_err_p95": q["gen_max_rel_err"].quantile(0.95),
    })


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check = reconstruction_check()
    plot_reconstruction_check(check)
    plot_price_profile_by_bar()
    check.round(4).to_csv(OUT_DIR / "real_reconstruction_check.csv")
    print(check.round(3).to_string())
    print(extraction_quality_summary().round(4).to_string())
