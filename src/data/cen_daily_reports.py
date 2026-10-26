"""Series horarias REALES del SEN desde los informes diarios públicos del Coordinador.

El Coordinador publica cada día un "Resumen Ejecutivo de Operación" en PDF. Sus
gráficos no son imágenes sino dibujos vectoriales, así que las líneas guardan
los valores exactos:

- **Costo marginal horario por barra** (USD/MWh). Hasta 2025 es el costo
  marginal *real preliminar* de 6 barras de 220 kV (Crucero, Cardones, Pan de
  Azúcar, Quillota, Charrúa, Puerto Montt), una polilínea de 24 puntos por
  barra. En los informes más recientes el gráfico pasa a ser el costo marginal
  *en línea*, cada 15 minutos (96 puntos), que aquí se promedia por hora.
- **Generación horaria por tecnología** (MW): áreas apiladas de térmica,
  hidráulica, solar, eólica y baterías. El grosor de cada área en cada hora es
  la generación de esa tecnología.

Ninguno de los dos gráficos trae una escala explícita confiable (la grilla está
ajustada a píxeles), así que cada día se calibra contra los números que el
mismo informe publica como texto:

- la leyenda del costo marginal da el **promedio diario** de cada barra: con 6
  u 8 barras y una recta de 2 parámetros, el ajuste queda sobredeterminado y su
  residuo mide la calidad de la reconstrucción;
- la tabla "Generación por Fuente" da la **energía diaria** (GWh) de cada
  tecnología: una sola escala (MW por punto) tiene que reproducir las cinco.

Los PDF se guardan en ``data/real/cen_reports/`` (no se versionan, ~1 GB para
2022-2025); las series extraídas sí se versionan.

    python -m src.data.cen_daily_reports download 2022-01-01 2025-12-31
    python -m src.data.cen_daily_reports extract
"""
from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = PROJECT_ROOT / "data" / "real"
PDF_DIR = REAL_DIR / "cen_reports"
CMG_HOURLY_PATH = REAL_DIR / "cen_cmg_hourly.csv.gz"
GEN_HOURLY_PATH = REAL_DIR / "cen_generation_hourly_by_tech.csv.gz"
REPORT_QUALITY_PATH = REAL_DIR / "cen_report_extraction_quality.csv"

BASE_URL = "https://www.coordinador.cl/wp-content/uploads"
SUFFIXES = ("", "-V1", "-V1-1", "-V2", "-1", "-V3")


def report_urls(day: date) -> list[str]:
    """URLs candidatas: el PDF se sube en la carpeta del mes del día o del siguiente, a veces con sufijo."""
    name = f"Resumen-Ejecutivo-de-Operacion-{day:%d-%m-%Y}"
    later = day + timedelta(days=3)
    months = [f"{day:%Y/%m}"] + ([f"{later:%Y/%m}"] if later.month != day.month else [])
    return [f"{BASE_URL}/{m}/{name}{s}.pdf" for m in months for s in SUFFIXES]


def download_report(day: date, pause: float = 1.0) -> Path | None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    local = PDF_DIR / f"{day.isoformat()}.pdf"
    if local.exists() and local.stat().st_size > 10_000:
        return local
    for url in report_urls(day):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            body = b""
        time.sleep(pause)
        if body[:4] == b"%PDF" and len(body) > 10_000:
            local.write_bytes(body)
            return local
    return None


def download_range(start: str, end: str) -> None:
    d, last = date.fromisoformat(start), date.fromisoformat(end)
    missing = []
    while d <= last:
        if download_report(d) is None:
            missing.append(d.isoformat())
        if d.day == 1:
            print(f"  {d:%Y-%m}  faltan hasta ahora: {len(missing)}", flush=True)
        d += timedelta(days=1)
    print(f"listo; {len(missing)} días sin informe: {missing[:20]}{' ...' if len(missing) > 20 else ''}")


# ---------------------------------------------------------------- extracción

TECHS = {"termica": r"T.rmica", "hidraulica": r"Hidr.ulica", "solar": r"Solar",
         "eolica": r"E.lica", "bess": r"Bess"}
# Colores de las áreas apiladas en los informes (varían levemente entre formatos).
TECH_COLORS = {
    "termica": [(0.89, 0.42, 0.04), (0.90, 0.42, 0.01)],
    "hidraulica": [(0.0, 0.44, 0.75), (0.0, 0.44, 0.76)],
    "solar": [(1.0, 1.0, 0.0)],
    "eolica": [(0.58, 0.80, 0.87), (0.58, 0.81, 0.87)],
    "bess": [(0.0, 0.69, 0.31), (0.24, 0.72, 0.27)],
}
BAR_NAMES = {"crucero": "Crucero", "cardones": "Cardones", "pan de az": "Pan de Azucar",
             "p.azucar": "Pan de Azucar", "quillota": "Quillota", "charr": "Charrua",
             "puerto montt": "Puerto Montt", "p.montt": "Puerto Montt", "tarapac": "Tarapaca",
             "atacama": "Atacama"}


def parse_number_cl(text: str) -> float:
    """'105.51' (2024) y '1.126,57' / '37,91' (formatos posteriores) -> float."""
    text = text.strip()
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)


def _bar_name(label: str) -> str | None:
    low = label.lower()
    for key, name in BAR_NAMES.items():
        if key in low:
            return name
    return None


NUMBER_RE = re.compile(r"^-?[0-9]{1,3}(?:\.[0-9]{3})*[.,][0-9]{2}$")


def parse_daily_table(words: list[tuple]) -> dict[str, float]:
    """GWh del día por tecnología desde la tabla "Generación por Fuente".

    ``words`` son las palabras de ``page.get_text("words")`` (x0, y0, x1, y1, texto, ...).
    La fila de cada tecnología empieza con su nombre, sigue con la energía del día
    y luego su participación en %; el eje del gráfico también empieza con el nombre
    pero no trae %, y la tabla de tiempo real viene en MWh (sobre 1.000).
    """
    out: dict[str, float] = {}
    for tech, pat in TECHS.items():
        heads = sorted((w for w in words if re.fullmatch(pat, w[4])), key=lambda w: (w[1], w[0]))
        for head in heads:
            row = sorted((w for w in words if abs(w[1] - head[1]) < 0.5 * (head[3] - head[1]) and w[0] > head[0]),
                         key=lambda w: w[0])
            if len(row) < 2 or not NUMBER_RE.match(row[0][4]) or not any("%" in w[4] for w in row[1:3]):
                continue
            val = parse_number_cl(row[0][4])
            if val < 1000:
                out[tech] = val
                break
    return out


def _nearest_tech(fill: tuple[float, float, float]) -> str | None:
    best, dist = None, 1e9
    for tech, cols in TECH_COLORS.items():
        for c in cols:
            d = sum(abs(a - b) for a, b in zip(fill, c))
            if d < dist:
                best, dist = tech, d
    return best if dist < 0.12 else None


def _polyline_points(d) -> np.ndarray:
    pts = [d["items"][0][1]] + [it[-1] for it in d["items"]]
    return np.array([(q.x, q.y) for q in pts])


def _subpaths(d, tol: float = 0.01) -> list[np.ndarray]:
    """Un área puede venir como varios subtrazos en un mismo path (las baterías
    descargan de madrugada y de noche, con un hueco al mediodía). Unirlos en un
    solo polígono crearía aristas que cruzan el hueco; se separan donde un
    segmento no empieza en el punto en que terminó el anterior."""
    parts, current = [], [d["items"][0][1]]
    for it in d["items"]:
        start, end = it[1], it[-1]
        if abs(start.x - current[-1].x) > tol or abs(start.y - current[-1].y) > tol:
            parts.append(current)
            current = [start]
        current.append(end)
    parts.append(current)
    return [np.array([(q.x, q.y) for q in part]) for part in parts if len(part) >= 3]


def vertical_span(polygon: np.ndarray, x: float, tol: float = 0.05) -> tuple[float, float] | None:
    """Cruce más alto y más bajo (y mínima, y máxima) de los bordes de un polígono
    cerrado con la vertical ``x``. No depende de que haya un vértice en cada hora
    (el PDF omite los puntos colineales de un tramo plano)."""
    ys = []
    closed = np.vstack([polygon, polygon[:1]])
    for (x0, y0), (x1, y1) in zip(closed[:-1], closed[1:]):
        lo, hi = min(x0, x1), max(x0, x1)
        if hi - lo < tol:
            if abs(x0 - x) < tol:
                ys += [y0, y1]
        elif lo - tol <= x <= hi + tol:
            ys.append(y0 + (y1 - y0) * (min(max(x, lo), hi) - x0) / (x1 - x0))
    return (float(min(ys)), float(max(ys))) if len(ys) >= 2 else None


def vertical_extent(polygon: np.ndarray, x: float) -> float:
    """Alto de un polígono cerrado en la vertical ``x`` (0 si no la cruza)."""
    span = vertical_span(polygon, x)
    return span[1] - span[0] if span else 0.0


def _cmg_lines(page) -> list[tuple[int, np.ndarray, object]]:
    """Polilíneas del gráfico de costo marginal (24 o 96 puntos, mitad derecha de la página)."""
    out = []
    for d in page.get_drawings():
        n = len(d["items"])
        if (d["color"] and d["fill"] is None and n in (23, 95)
                and all(it[0] == "l" for it in d["items"]) and d["rect"].x0 > page.rect.width * 0.38):
            r, g, b = (int(round(v * 255)) for v in d["color"])
            out.append(((r << 16) | (g << 8) | b, _polyline_points(d), d["rect"]))
    return out


def _legend(page) -> list[tuple[int, float, str, float]]:
    """(color, y, barra, promedio) de las leyendas "Promedio:" del costo marginal."""
    entries, pending = [], None
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                text = span["text"].strip()
                if span["color"] != 0 and 0 < len(text) <= 2:
                    pending = (span["color"], span["bbox"][1])
                    continue
                if pending is not None:
                    m = re.match(r"(-?[0-9][0-9.,]*)\s+(.+)", text)
                    if m and _bar_name(m.group(2)):
                        try:
                            entries.append((pending[0], pending[1], _bar_name(m.group(2)), parse_number_cl(m.group(1))))
                        except ValueError:
                            pass
                pending = None
    return entries


def _close(a: int, b: int, tol: int = 8) -> bool:
    return all(abs(((a >> s) & 255) - ((b >> s) & 255)) <= tol for s in (16, 8, 0))


def extract_cmg(doc) -> tuple[pd.DataFrame, float, str]:
    """Costo marginal horario por barra, calibrado contra los promedios diarios de la leyenda."""
    for page in doc:
        lines = _cmg_lines(page)
        legend = _legend(page)
        if len(lines) < 3 or len(legend) < 3:
            continue
        # En los informes recientes hay dos gráficos con los mismos colores (en
        # línea arriba, programado abajo): por cada color se toma la línea y la
        # entrada de leyenda que están más arriba en la página.
        pairs = []
        for col in {c for c, _, _ in lines}:
            line = min((ln for ln in lines if ln[0] == col), key=lambda ln: ln[1][:, 1].mean())
            hits = [e for e in legend if _close(e[0], col)]
            if hits:
                entry = min(hits, key=lambda e: e[1])
                pairs.append((entry[2], entry[3], line[1][:, 1]))
        if len(pairs) < 3:
            continue
        X = np.array([p[2].mean() for p in pairs])
        Y = np.array([p[1] for p in pairs])
        slope, icpt = np.polyfit(X, Y, 1)
        resid = float(np.abs(Y - (icpt + slope * X)).max())
        kind = "real_preliminar" if len(pairs[0][2]) == 24 else "en_linea"
        rows = []
        for bar, _, ys in pairs:
            vals = icpt + slope * ys
            if len(vals) == 96:
                vals = vals.reshape(24, 4).mean(axis=1)
            for h, v in enumerate(vals, start=1):
                rows.append((bar, h, round(float(v), 2)))
        return pd.DataFrame(rows, columns=["bar", "hour", "cmg_usd_mwh"]), resid, kind
    return pd.DataFrame(columns=["bar", "hour", "cmg_usd_mwh"]), float("nan"), "none"


def extract_generation(doc, table: dict[str, float]) -> tuple[pd.DataFrame, float]:
    """Generación horaria por tecnología (MW) desde las áreas apiladas, escalada con la tabla diaria."""
    for page in doc:
        areas, top_lines = [], []
        for d in page.get_drawings():
            f = d["fill"]
            r = d["rect"]
            if r.x1 >= page.rect.width * 0.4 or r.y1 >= page.rect.height * 0.45:
                continue
            if f and len(d["items"]) >= 20:
                tech = _nearest_tech(tuple(f))
                if tech:
                    areas.extend((tech, part) for part in _subpaths(d))
            elif d["color"] == (0.0, 0.0, 0.0) and f is None and len(d["items"]) == 23:
                top_lines.append(_polyline_points(d))
        if len(areas) < 3:
            continue
        grid = None
        for _, P in areas:
            xr = np.round(P[:, 0], 1)
            if len(np.unique(xr)) == 24:
                grid = np.array([P[xr == v, 0].mean() for v in np.unique(xr)])  # x exactas de cada hora
                break
        if grid is None:
            continue
        thick: dict[str, np.ndarray] = {}
        for tech, P in areas:
            thick[tech] = thick.get(tech, 0) + np.array([vertical_extent(P, x) for x in grid])
        if "eolica" not in thick and top_lines:
            # Informes de comienzos de 2022: la eólica, arriba del apilado, se
            # dibuja solo como una línea negra; su grosor es la distancia entre
            # esa línea y el techo de las demás áreas.
            line = min(top_lines, key=lambda L: L[:, 1].mean())
            stack_top = np.array([min((sp[0] for P in (a[1] for a in areas) if (sp := vertical_span(P, x))),
                                      default=np.nan) for x in grid])
            line_y = np.interp(grid, line[:, 0], line[:, 1])
            thick["eolica"] = np.clip(np.nan_to_num(stack_top - line_y), 0, None)
        common = [t for t in thick if t in table and thick[t].sum() > 0]
        # La escala se mide con las cuatro tecnologías grandes: el área de las
        # baterías (1-2% de la energía) se dibuja de forma inconsistente entre
        # formatos y no sirve para calibrar; igual se ajusta a su energía diaria.
        scale_techs = [t for t in common if t != "bess"]
        if not scale_techs:
            continue
        E = np.array([table[t] * 1000 for t in scale_techs])     # MWh del día
        T = np.array([thick[t].sum() for t in scale_techs])       # puntos·hora
        scale = float((E * T).sum() / (T * T).sum())              # MW por punto
        err = float(np.max(np.abs(scale * T - E) / np.maximum(E, 1000)))
        # La forma horaria sale del gráfico; la energía de cada tecnología se ajusta a
        # la de la tabla del mismo informe (en los informes con áreas suavizadas una
        # sola escala no las reproduce todas, ver ``gen_max_rel_err``).
        factor = {t: (table[t] * 1000) / (scale * thick[t].sum()) for t in common}
        rows = [(tech, h, round(float(v * scale * factor.get(tech, 1.0)), 1))
                for tech, arr in thick.items() for h, v in enumerate(arr, 1)]
        return pd.DataFrame(rows, columns=["tech", "hour", "mw"]), err
    return pd.DataFrame(columns=["tech", "hour", "mw"]), float("nan")


def extract_report(path: Path) -> dict:
    import pymupdf

    doc = pymupdf.open(path)
    table: dict[str, float] = {}
    for page in doc:
        found = parse_daily_table(page.get_text("words"))
        if len(found) > len(table):
            table = found
    cmg, cmg_resid, kind = extract_cmg(doc)
    gen, gen_err = extract_generation(doc, table)
    return {"cmg": cmg, "cmg_resid": cmg_resid, "cmg_kind": kind, "gen": gen, "gen_err": gen_err, "table": table}


def extract_all() -> None:
    cmg_parts, gen_parts, quality = [], [], []
    for path in sorted(PDF_DIR.glob("*.pdf")):
        day = path.stem
        try:
            r = extract_report(path)
        except Exception as exc:  # un PDF corrupto no detiene la corrida
            quality.append({"date": day, "error": repr(exc)[:120]})
            continue
        if len(r["cmg"]):
            cmg_parts.append(r["cmg"].assign(date=day, kind=r["cmg_kind"]))
        if len(r["gen"]):
            gen_parts.append(r["gen"].assign(date=day))
        quality.append({"date": day, "cmg_kind": r["cmg_kind"], "cmg_bars": r["cmg"]["bar"].nunique() if len(r["cmg"]) else 0,
                        "cmg_max_resid_usd": r["cmg_resid"], "gen_max_rel_err": r["gen_err"],
                        **{f"table_{k}_gwh": v for k, v in r["table"].items()}})
    cmg = pd.concat(cmg_parts)[["date", "hour", "bar", "kind", "cmg_usd_mwh"]]
    gen = pd.concat(gen_parts).pivot_table(index=["date", "hour"], columns="tech", values="mw").reset_index()
    cmg.to_csv(CMG_HOURLY_PATH, index=False)
    gen.to_csv(GEN_HOURLY_PATH, index=False)
    pd.DataFrame(quality).to_csv(REPORT_QUALITY_PATH, index=False)
    print(f"cmg: {cmg['date'].nunique()} días, gen: {gen['date'].nunique()} días -> {REAL_DIR}")


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download_range(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "extract":
        extract_all()
