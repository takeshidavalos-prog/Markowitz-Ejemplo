"""
Optimizador de Cartera de Arrendamiento Financiero — Frontera Eficiente (Markowitz)
=====================================================================================

Aplicación Streamlit que aplica la Teoría Moderna de Portafolios (Markowitz)
a segmentos de cartera de arrendamiento financiero, definidos como la
combinación sector económico x región geográfica.

Todo el código vive en un único archivo (app.py) a propósito: al desplegar
en Streamlit Community Cloud (o subir el repo manualmente vía la interfaz
web de GitHub) es común que una carpeta de módulos (por ejemplo `src/`) no
se suba completa y la app truene con `ModuleNotFoundError`. Al no depender
de ningún paquete interno, ese riesgo desaparece.

Ejecutar localmente:
    streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog, minimize

# =============================================================================
# SECCIÓN 1 — Esquema de columnas esperado
# =============================================================================
# Cualquier archivo (sintético o real) que respete estos nombres y su
# significado puede utilizarse en la aplicación sin modificar el código.

COLUMNAS_REQUERIDAS = [
    "fecha",
    "sector",
    "region",
    "segmento",
    "exposicion",
    "num_contratos",
    "tasa_bruta_mensual",
    "pd_mensual",
    "lgd",
    "perdida_esperada_monto",
    "perdida_esperada_tasa",
    "rendimiento_neto_mensual",
    "pct_cartera_mensual",
    "limite_maximo_ejemplo_pct",
]

COLUMNAS_NUMERICAS = [
    "exposicion",
    "num_contratos",
    "tasa_bruta_mensual",
    "pd_mensual",
    "lgd",
    "perdida_esperada_monto",
    "perdida_esperada_tasa",
    "rendimiento_neto_mensual",
    "pct_cartera_mensual",
    "limite_maximo_ejemplo_pct",
]


# =============================================================================
# SECCIÓN 2 — Carga, validación y transformación de datos
# =============================================================================
class ErrorEstructuraDatos(Exception):
    """Se lanza cuando el archivo cargado no respeta el esquema esperado."""


@dataclass
class ResultadoCarga:
    df: pd.DataFrame
    advertencias: list[str]


def cargar_datos(fuente) -> ResultadoCarga:
    """
    Carga un archivo CSV (ruta, buffer de Streamlit, o BytesIO) y valida su
    estructura contra el esquema esperado.
    """
    advertencias: list[str] = []

    try:
        if hasattr(fuente, "read"):
            contenido = fuente.read()
            if isinstance(contenido, bytes):
                buffer = io.BytesIO(contenido)
            else:
                buffer = io.StringIO(contenido)
            df = pd.read_csv(buffer, encoding="utf-8-sig")
        else:
            df = pd.read_csv(fuente, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        raise ErrorEstructuraDatos(f"No fue posible leer el archivo CSV: {exc}") from exc

    df.columns = [c.strip() for c in df.columns]

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ErrorEstructuraDatos(
            "El archivo no tiene la estructura de columnas esperada. "
            f"Faltan las columnas: {', '.join(faltantes)}."
        )

    sobrantes = [c for c in df.columns if c not in COLUMNAS_REQUERIDAS]
    if sobrantes:
        advertencias.append(
            f"Se encontraron columnas adicionales no utilizadas por el modelo: {', '.join(sobrantes)}."
        )

    df = df[COLUMNAS_REQUERIDAS].copy()

    try:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise ErrorEstructuraDatos(
            f"La columna 'fecha' contiene valores que no se pudieron interpretar como fecha: {exc}"
        ) from exc

    for col in COLUMNAS_NUMERICAS:
        antes = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        despues_nan = df[col].isna().sum()
        if despues_nan > antes:
            advertencias.append(
                f"La columna '{col}' tenía valores no numéricos que fueron convertidos a NaN "
                f"({despues_nan - antes} celda(s))."
            )

    for col in ("sector", "region", "segmento"):
        df[col] = df[col].astype(str).str.strip()

    filas_antes = len(df)
    df = df.dropna(subset=["fecha", "segmento", "rendimiento_neto_mensual"])
    filas_despues = len(df)
    if filas_despues < filas_antes:
        advertencias.append(
            f"Se descartaron {filas_antes - filas_despues} fila(s) con fecha, segmento o "
            "rendimiento_neto_mensual faltante."
        )

    duplicados = df.duplicated(subset=["fecha", "segmento"]).sum()
    if duplicados > 0:
        advertencias.append(
            f"Se encontraron {duplicados} registro(s) duplicados de fecha+segmento; se "
            "promediará su valor al construir las series."
        )

    df = df.sort_values(["fecha", "sector", "region"]).reset_index(drop=True)

    return ResultadoCarga(df=df, advertencias=advertencias)


def rango_fechas(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["fecha"].min(), df["fecha"].max()


def lista_segmentos(df: pd.DataFrame) -> list[str]:
    return sorted(df["segmento"].unique().tolist())


def filtrar(df: pd.DataFrame, fecha_inicio, fecha_fin, segmentos: list[str]) -> pd.DataFrame:
    mask = (
        (df["fecha"] >= pd.Timestamp(fecha_inicio))
        & (df["fecha"] <= pd.Timestamp(fecha_fin))
        & (df["segmento"].isin(segmentos))
    )
    return df.loc[mask].copy()


def matriz_rendimientos(df: pd.DataFrame) -> pd.DataFrame:
    """Pivotea el histórico a una matriz fecha x segmento de rendimiento_neto_mensual."""
    tabla = df.pivot_table(
        index="fecha", columns="segmento", values="rendimiento_neto_mensual", aggfunc="mean"
    )
    return tabla.sort_index()


def distribucion_actual(df: pd.DataFrame, segmentos: list[str]) -> pd.Series:
    """Distribución de la cartera actual (pct_cartera_mensual) del mes más reciente, renormalizada a 1."""
    fecha_max = df["fecha"].max()
    ultimo = df.loc[df["fecha"] == fecha_max].groupby("segmento")["pct_cartera_mensual"].mean()
    ultimo = ultimo.reindex(segmentos).fillna(0.0)
    total = ultimo.sum()
    if total > 0:
        ultimo = ultimo / total
    return ultimo


def exposicion_actual(df: pd.DataFrame, segmentos: list[str]) -> pd.Series:
    fecha_max = df["fecha"].max()
    ultimo = df.loc[df["fecha"] == fecha_max].groupby("segmento")["exposicion"].sum()
    return ultimo.reindex(segmentos).fillna(0.0)


def limites_maximos_ejemplo(df: pd.DataFrame, segmentos: list[str]) -> pd.Series:
    """Toma el límite máximo de ejemplo/plantilla más reciente por segmento."""
    fecha_max = df["fecha"].max()
    ultimo = df.loc[df["fecha"] == fecha_max].groupby("segmento")["limite_maximo_ejemplo_pct"].mean()
    ultimo = ultimo.reindex(segmentos)
    media_global = ultimo.mean() if not np.isnan(ultimo.mean()) else 1.0
    return ultimo.fillna(media_global)


def resumen_calidad(df: pd.DataFrame) -> dict:
    return {
        "n_filas": len(df),
        "n_segmentos": df["segmento"].nunique(),
        "n_sectores": df["sector"].nunique(),
        "n_regiones": df["region"].nunique(),
        "fecha_min": df["fecha"].min(),
        "fecha_max": df["fecha"].max(),
    }


# =============================================================================
# SECCIÓN 3 — Modelo de optimización (Teoría Moderna de Portafolios / Markowitz)
# =============================================================================
MESES_POR_ANIO = 12


@dataclass
class Estadisticos:
    mu: pd.Series
    cov: pd.DataFrame
    corr: pd.DataFrame
    sigma: pd.Series
    anualizado: bool


def calcular_estadisticos(matriz_rend: pd.DataFrame, anualizar: bool = False) -> Estadisticos:
    """Retorno esperado, matriz de covarianza/correlación y volatilidad por segmento."""
    matriz = matriz_rend.dropna(axis=0, how="any")
    if matriz.shape[0] < 3:
        matriz = matriz_rend.fillna(matriz_rend.mean())

    mu = matriz.mean(axis=0)
    cov = matriz.cov()
    corr = matriz.corr()
    sigma = matriz.std(axis=0, ddof=1)

    if anualizar:
        mu = mu * MESES_POR_ANIO
        cov = cov * MESES_POR_ANIO
        sigma = sigma * np.sqrt(MESES_POR_ANIO)

    return Estadisticos(mu=mu, cov=cov, corr=corr, sigma=sigma, anualizado=anualizar)


def rendimiento_portafolio(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(w, mu))


def riesgo_portafolio(w: np.ndarray, cov: np.ndarray) -> float:
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)))


def hhi(w: np.ndarray) -> float:
    return float(np.sum(np.square(w)))


def ratio_retorno_riesgo(r_p: float, sigma_p: float, r_referencia: float) -> float:
    if sigma_p is None or sigma_p <= 1e-12:
        return float("nan")
    return (r_p - r_referencia) / sigma_p


@dataclass
class ResultadoOptimizacion:
    exito: bool
    pesos: np.ndarray | None
    retorno: float | None
    riesgo: float | None
    mensaje: str = ""


def _bounds(n: int, limites_min: np.ndarray, limites_max: np.ndarray):
    return [(float(limites_min[i]), float(limites_max[i])) for i in range(n)]


def rango_retorno_factible(mu: np.ndarray, limites_min: np.ndarray, limites_max: np.ndarray):
    """Retorno esperado mínimo/máximo alcanzable dado sum(w)=1 y límites min/max (vía LP)."""
    n = len(mu)
    bounds = _bounds(n, limites_min, limites_max)
    a_eq = [np.ones(n)]
    b_eq = [1.0]

    res_max = linprog(c=-mu, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    res_min = linprog(c=mu, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res_max.success or not res_min.success:
        return None, None

    return float(res_min.fun), float(-res_max.fun)


def optimizar_min_varianza(
    mu: np.ndarray,
    cov: np.ndarray,
    limites_min: np.ndarray,
    limites_max: np.ndarray,
    retorno_objetivo: float | None = None,
) -> ResultadoOptimizacion:
    """Minimiza la varianza sujeto a sum(w)=1, límites min/max y, opcionalmente, w^T mu = R*."""
    n = len(mu)
    w0 = np.repeat(1.0 / n, n)
    bounds = _bounds(n, limites_min, limites_max)

    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if retorno_objetivo is not None:
        restricciones.append({"type": "eq", "fun": lambda w, r=retorno_objetivo: w @ mu - r})

    resultado = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=restricciones,
        options={"maxiter": 1000, "ftol": 1e-14},
    )

    if not resultado.success:
        return ResultadoOptimizacion(False, None, None, None, resultado.message)

    w = np.clip(resultado.x, 0, None)
    w = w / w.sum()
    return ResultadoOptimizacion(True, w, rendimiento_portafolio(w, mu), riesgo_portafolio(w, cov), "ok")


def optimizar_max_utilidad(
    mu: np.ndarray,
    cov: np.ndarray,
    limites_min: np.ndarray,
    limites_max: np.ndarray,
    aversion_riesgo: float,
) -> ResultadoOptimizacion:
    """Maximiza U(w) = w^T mu - (lambda/2) * w^T Sigma w sujeto a sum(w)=1 y límites por segmento."""
    n = len(mu)
    w0 = np.repeat(1.0 / n, n)
    bounds = _bounds(n, limites_min, limites_max)
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objetivo(w):
        return -(w @ mu - 0.5 * aversion_riesgo * (w @ cov @ w))

    resultado = minimize(
        objetivo,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=restricciones,
        options={"maxiter": 1000, "ftol": 1e-14},
    )

    if not resultado.success:
        return ResultadoOptimizacion(False, None, None, None, resultado.message)

    w = np.clip(resultado.x, 0, None)
    w = w / w.sum()
    return ResultadoOptimizacion(True, w, rendimiento_portafolio(w, mu), riesgo_portafolio(w, cov), "ok")


@dataclass
class FronteraEficiente:
    retornos: np.ndarray
    riesgos: np.ndarray
    pesos: list
    r_gmv: float          # retorno del portafolio de mínima varianza global (ancla de la frontera)
    sigma_gmv: float       # riesgo del portafolio de mínima varianza global
    r_max_factible: float


def calcular_frontera_eficiente(
    mu: np.ndarray,
    cov: np.ndarray,
    limites_min: np.ndarray,
    limites_max: np.ndarray,
    n_puntos: int = 40,
) -> FronteraEficiente | None:
    """
    Traza la frontera eficiente resolviendo mínima varianza para una malla de
    retornos objetivo.

    Importante: para cada retorno objetivo R* existe una única cartera de
    varianza mínima, pero esa relación riesgo-retorno NO es monótona en todo
    el rango factible — es una parábola ("bala de Markowitz") con vértice en
    el portafolio de mínima varianza global (GMV). Solo la rama superior de
    esa parábola (retornos >= retorno del GMV) es la verdadera "frontera
    eficiente"; la rama inferior son carteras dominadas (mismo riesgo, menor
    retorno que otra cartera factible). Si se grafican ambas ramas ordenando
    únicamente por riesgo, la línea salta de una rama a otra y se ve como un
    zigzag en vez de una curva suave. Por eso aquí se calcula primero el GMV
    y solo se barre la malla de retornos objetivo desde ese punto hacia el
    retorno máximo factible.
    """
    r_min, r_max = rango_retorno_factible(mu, limites_min, limites_max)
    if r_min is None:
        return None

    res_gmv = optimizar_min_varianza(mu, cov, limites_min, limites_max, retorno_objetivo=None)
    if not res_gmv.exito:
        return None
    r_gmv, sigma_gmv = res_gmv.retorno, res_gmv.riesgo

    if r_max <= r_gmv:
        # Caso degenerado (prácticamente un solo portafolio factible)
        return FronteraEficiente(
            np.array([r_gmv]), np.array([sigma_gmv]), [res_gmv.pesos], r_gmv, sigma_gmv, r_max
        )

    eps = (r_max - r_gmv) * 0.01
    objetivos = np.linspace(r_gmv, r_max - eps, max(n_puntos - 1, 2))

    retornos, riesgos, pesos = [r_gmv], [sigma_gmv], [res_gmv.pesos]
    for r_obj in objetivos[1:]:
        res = optimizar_min_varianza(mu, cov, limites_min, limites_max, retorno_objetivo=r_obj)
        if res.exito:
            retornos.append(res.retorno)
            riesgos.append(res.riesgo)
            pesos.append(res.pesos)

    orden = np.argsort(riesgos)
    retornos = np.array(retornos)[orden]
    riesgos = np.array(riesgos)[orden]
    pesos = [pesos[i] for i in orden]

    return FronteraEficiente(retornos, riesgos, pesos, r_gmv, sigma_gmv, r_max)


def simulacion_portafolios_aleatorios(
    mu: np.ndarray,
    cov: np.ndarray,
    limites_min: np.ndarray,
    limites_max: np.ndarray,
    n_sim: int = 3000,
    semilla: int | None = 42,
):
    """Simulación Monte Carlo de portafolios aleatorios factibles (rechazo por muestreo Dirichlet)."""
    n = len(mu)
    rng = np.random.default_rng(semilla)
    pesos_validos = []

    max_intentos = max(n_sim * 60, 20000)
    intentos = 0
    alpha = np.ones(n) * 2.0

    while len(pesos_validos) < n_sim and intentos < max_intentos:
        intentos += 1
        lote = rng.dirichlet(alpha, size=200)
        cumple = np.all(lote >= limites_min - 1e-9, axis=1) & np.all(lote <= limites_max + 1e-9, axis=1)
        for w in lote[cumple]:
            pesos_validos.append(w)
            if len(pesos_validos) >= n_sim:
                break

    if not pesos_validos:
        return None

    pesos = np.array(pesos_validos)
    pesos = pesos / pesos.sum(axis=1, keepdims=True)
    retornos = pesos @ mu
    riesgos = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos))

    return pesos, retornos, riesgos


# =============================================================================
# SECCIÓN 4 — Estilo visual (paleta azul/gris tipo fintech, tipografía Tahoma)
# =============================================================================
COLOR_PRIMARY = "#0B4DA2"
COLOR_PRIMARY_DARK = "#062F63"
COLOR_PRIMARY_LIGHT = "#DCE7F7"
COLOR_ACCENT = "#3E7CC2"
COLOR_GRAY = "#6B7280"
COLOR_GRAY_LIGHT = "#E7E9EC"
COLOR_BG = "#F3F5F8"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#14181F"
COLOR_POSITIVE = "#1E8E5A"
COLOR_NEGATIVE = "#C0392B"

FUENTE = "Tahoma, Geneva, Verdana, sans-serif"

ESCALA_CORRELACION = [
    [0.0, "#C0392B"],
    [0.25, "#E7E9EC"],
    [0.5, "#DCE7F7"],
    [0.75, "#3E7CC2"],
    [1.0, "#062F63"],
]


def css_global() -> str:
    # NOTA sobre robustez: a propósito NO se usa el selector [class*="css"]
    # (coincide con casi cualquier elemento que Streamlit dibuja, incluidos
    # los íconos tipográficos de flechas/checks/cerrar) ni se fuerza
    # `font-family` con !important sobre selectores amplios como `*`. Hacerlo
    # sobrescribe también los íconos-de-fuente (Material Symbols) que usa
    # Streamlit internamente, y el glifo del ícono se reemplaza por un
    # carácter Tahoma "roto"/una caja — eso es lo que se veía como texto o
    # símbolos extraños sin origen claro. Por eso aquí solo se fuerza
    # `color` (que no rompe glifos) y `font-family` se aplica sin
    # !important sobre selectores de texto específicos, donde las reglas
    # propias de Streamlit para íconos (más específicas) siguen ganando.
    return f"""
    <style>
        html, body, .stMarkdown, .stText, .stDataFrame, .stTable,
        .stMetric, .stButton, .stSelectbox, .stMultiSelect, .stSlider, .stRadio,
        .stNumberInput, .stDateInput, .stTextInput {{
            font-family: {FUENTE};
            color: {COLOR_TEXT};
        }}

        .stApp {{
            background-color: {COLOR_BG};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLOR_PRIMARY_DARK};
            border-right: 1px solid {COLOR_GRAY_LIGHT};
        }}
        section[data-testid="stSidebar"] * {{
            color: #F3F5F8 !important;
        }}
        /* Los campos de texto/fecha/número del sidebar (date_input, number_input,
           el buscador del multiselect) se dibujan con fondo BLANCO propio de
           Streamlit, sin importar el fondo oscuro del sidebar. Como la regla de
           arriba fuerza texto casi blanco en TODO el sidebar, ese texto quedaba
           blanco sobre blanco (invisible). Aquí se revierte el color solo para
           los <input> reales, dejando intactas las etiquetas y los "chips" de
           segmentos seleccionados (que sí tienen fondo oscuro/azul y sí deben
           verse en texto claro). */
        section[data-testid="stSidebar"] input {{
            color: {COLOR_TEXT} !important;
            background-color: #FFFFFF;
        }}
        /* El selector de rango de fechas no usa un <input> visible para el
           valor (usa "spinbuttons" React accesibles dentro de un contenedor
           con fondo blanco, con el <input> real oculto para lectores de
           pantalla), así que la regla de arriba no lo alcanza: se corrige
           aparte, por especificidad, apuntando directo a su contenedor. */
        section[data-testid="stSidebar"] [data-testid="stDateInputField"],
        section[data-testid="stSidebar"] [data-testid="stDateInputField"] * {{
            color: {COLOR_TEXT} !important;
        }}
        /* Texto con formato "código" en markdown (ej. `limite_maximo_ejemplo_pct`)
           también trae su propio fondo gris claro — mismo problema de texto
           claro sobre fondo claro que las otras reglas de esta sección. */
        section[data-testid="stSidebar"] code {{
            color: {COLOR_PRIMARY_DARK} !important;
            background-color: #EEF2F8;
        }}
        /* El recuadro de "Cargar archivo CSV" (dropzone) también tiene fondo
           claro propio — mismo problema: el botón "Upload" y el texto
           "200MB per file • CSV" quedaban en blanco sobre ese fondo claro.
           Se corrige igual que los demás casos, apuntando directo a sus
           contenedores por especificidad. Una vez cargado un archivo, la fila
           con su nombre también usa fondo claro, por eso se cubre todo
           [data-testid^="stFileUploader"]. */
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
        section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {{
            color: {COLOR_TEXT} !important;
        }}
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div div div {{
            background-color: {COLOR_ACCENT} !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.15);
        }}

        h1, h2, h3, h4 {{
            font-family: {FUENTE} !important;
            color: {COLOR_PRIMARY_DARK} !important;
            font-weight: 700 !important;
            letter-spacing: 0.2px;
        }}

        .app-header {{
            background: linear-gradient(90deg, {COLOR_PRIMARY_DARK} 0%, {COLOR_PRIMARY} 100%);
            padding: 22px 28px;
            border-radius: 10px;
            margin-bottom: 18px;
            box-shadow: 0 2px 10px rgba(6,47,99,0.18);
        }}
        .app-header h1 {{
            color: #FFFFFF !important;
            margin: 0 0 4px 0;
            font-size: 1.55rem !important;
        }}
        .app-header p {{
            color: #DCE7F7 !important;
            margin: 0;
            font-size: 0.92rem;
        }}

        .section-title {{
            font-weight: 700;
            color: {COLOR_PRIMARY_DARK};
            font-size: 1.05rem;
            border-left: 4px solid {COLOR_PRIMARY};
            padding-left: 10px;
            margin-bottom: 10px;
        }}

        div[data-testid="stMetric"] {{
            background-color: {COLOR_CARD};
            border: 1px solid {COLOR_GRAY_LIGHT};
            border-left: 4px solid {COLOR_PRIMARY};
            border-radius: 8px;
            padding: 12px 14px 8px 14px;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {COLOR_GRAY} !important;
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {COLOR_PRIMARY_DARK} !important;
            font-weight: 700 !important;
            font-size: 1.55rem !important;
            overflow-wrap: anywhere;
            white-space: normal !important;
        }}

        .stButton>button, .stDownloadButton>button {{
            background-color: {COLOR_PRIMARY};
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            font-family: {FUENTE} !important;
            font-weight: 600;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            background-color: {COLOR_PRIMARY_DARK};
            color: #FFFFFF;
        }}

        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
    </style>
    """


def layout_plotly(fig, altura=None, leyenda=True):
    """Aplica un layout consistente tipo terminal financiera a una figura Plotly."""
    fig.update_layout(
        font=dict(family=FUENTE, color=COLOR_TEXT, size=12),
        plot_bgcolor=COLOR_CARD,
        paper_bgcolor=COLOR_CARD,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=leyenda,
        legend=dict(font=dict(family=FUENTE, size=11)),
        hoverlabel=dict(font_family=FUENTE, bgcolor=COLOR_PRIMARY_DARK, font_color="white"),
    )
    fig.update_xaxes(gridcolor=COLOR_GRAY_LIGHT, zerolinecolor=COLOR_GRAY_LIGHT)
    fig.update_yaxes(gridcolor=COLOR_GRAY_LIGHT, zerolinecolor=COLOR_GRAY_LIGHT)
    if altura:
        fig.update_layout(height=altura)
    return fig


# =============================================================================
# SECCIÓN 5 — Datos de ejemplo (generados en memoria, sin depender de ningún
# archivo dentro del repositorio)
# =============================================================================
# NOTA: el dataset de ejemplo se genera aquí mismo, en memoria, con la misma
# lógica y semilla que data/generar_datos_sinteticos.py, en vez de leerse
# desde data/base_datos_cartera_sintetica.csv. Esto es deliberado: si esa
# carpeta no llega completa al repositorio de GitHub (algo común al subir
# archivos manualmente desde la interfaz web en vez de usar `git push`), la
# app dejaba de poder mostrar una demo funcional. Generándolo en memoria, la
# app SIEMPRE tiene datos de ejemplo disponibles, sin importar qué archivos
# hayan llegado al repositorio. El CSV en data/ se conserva solo como
# referencia legible y para quien quiera regenerarlo por su cuenta.

_SECTORES_EJEMPLO = ["Manufactura", "Comercio", "Servicios", "Transporte", "Construccion", "Agroindustria"]
_REGIONES_EJEMPLO = ["Norte", "Centro", "Bajio", "Sur"]

_PARAMS_SECTOR_EJEMPLO = {
    "Manufactura":   dict(tasa_base=0.0165, pd_base=0.0060, lgd_base=0.40, tam_base=180_000_000),
    "Comercio":      dict(tasa_base=0.0180, pd_base=0.0090, lgd_base=0.45, tam_base=150_000_000),
    "Servicios":     dict(tasa_base=0.0155, pd_base=0.0050, lgd_base=0.38, tam_base=120_000_000),
    "Transporte":    dict(tasa_base=0.0195, pd_base=0.0110, lgd_base=0.50, tam_base=110_000_000),
    "Construccion":  dict(tasa_base=0.0210, pd_base=0.0130, lgd_base=0.55, tam_base=95_000_000),
    "Agroindustria": dict(tasa_base=0.0185, pd_base=0.0100, lgd_base=0.48, tam_base=80_000_000),
}

_AJUSTE_REGION_EJEMPLO = {
    "Norte":  dict(tasa_mult=1.05, pd_mult=0.95, tam_mult=1.15),
    "Centro": dict(tasa_mult=1.00, pd_mult=1.00, tam_mult=1.30),
    "Bajio":  dict(tasa_mult=0.98, pd_mult=0.90, tam_mult=1.00),
    "Sur":    dict(tasa_mult=1.10, pd_mult=1.20, tam_mult=0.70),
}

_TAM_CONTRATO_PROM_EJEMPLO = {
    "Manufactura": 2_800_000, "Comercio": 1_200_000, "Servicios": 900_000,
    "Transporte": 1_600_000, "Construccion": 3_500_000, "Agroindustria": 2_200_000,
}


def _generar_datos_ejemplo(semilla: int = 42) -> pd.DataFrame:
    """
    Genera en memoria la misma base sintética de ejemplo que produce
    data/generar_datos_sinteticos.py (24 segmentos = 6 sectores x 4 regiones,
    48 meses de historia), sin depender de ningún archivo en disco.
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(end="2024-12-31", periods=48, freq="ME")
    limite_min, limite_max = 0.08, 0.22
    registros = []

    for sector in _SECTORES_EJEMPLO:
        ps = _PARAMS_SECTOR_EJEMPLO[sector]
        for region in _REGIONES_EJEMPLO:
            ar = _AJUSTE_REGION_EJEMPLO[region]
            segmento = f"{sector} - {region}"

            tasa_media = ps["tasa_base"] * ar["tasa_mult"]
            pd_media = ps["pd_base"] * ar["pd_mult"]
            lgd_media = ps["lgd_base"]
            tam_medio = ps["tam_base"] * ar["tam_mult"]
            limite_pct = rng.uniform(limite_min, limite_max)

            n = len(fechas)

            ruido_tasa = rng.normal(0, tasa_media * 0.08, n)
            tasa_bruta = np.clip(
                tasa_media + np.cumsum(ruido_tasa) * 0.15 / np.sqrt(n) + rng.normal(0, tasa_media * 0.05, n),
                0.004,
                0.05,
            )

            pd_serie = np.zeros(n)
            pd_serie[0] = pd_media
            for t in range(1, n):
                choque = rng.normal(0, pd_media * 0.25)
                pd_serie[t] = 0.85 * pd_serie[t - 1] + 0.15 * pd_media + choque
            pd_serie = np.clip(pd_serie, 0.0005, 0.08)

            lgd_serie = np.clip(lgd_media + rng.normal(0, 0.03, n), 0.15, 0.85)

            tendencia = np.linspace(0, rng.uniform(-0.10, 0.35), n)
            estacional = 0.03 * np.sin(np.linspace(0, 4 * np.pi, n))
            ruido_exp = rng.normal(0, 0.04, n)
            exposicion = tam_medio * (1 + tendencia + estacional + np.cumsum(ruido_exp) * 0.05)
            exposicion = np.clip(exposicion, tam_medio * 0.3, None)

            tam_contrato_prom = _TAM_CONTRATO_PROM_EJEMPLO[sector]
            num_contratos = np.maximum(
                5, np.round(exposicion / tam_contrato_prom * rng.uniform(0.85, 1.15, n))
            ).astype(int)

            perdida_esperada_tasa = pd_serie * lgd_serie
            perdida_esperada_monto = perdida_esperada_tasa * exposicion
            rendimiento_neto_mensual = tasa_bruta - perdida_esperada_tasa

            for i, fecha in enumerate(fechas):
                registros.append(
                    dict(
                        fecha=fecha.strftime("%Y-%m-%d"),
                        sector=sector,
                        region=region,
                        segmento=segmento,
                        exposicion=round(float(exposicion[i]), 2),
                        num_contratos=int(num_contratos[i]),
                        tasa_bruta_mensual=round(float(tasa_bruta[i]), 6),
                        pd_mensual=round(float(pd_serie[i]), 6),
                        lgd=round(float(lgd_serie[i]), 6),
                        perdida_esperada_monto=round(float(perdida_esperada_monto[i]), 2),
                        perdida_esperada_tasa=round(float(perdida_esperada_tasa[i]), 6),
                        rendimiento_neto_mensual=round(float(rendimiento_neto_mensual[i]), 6),
                        limite_maximo_ejemplo_pct=round(float(limite_pct), 4),
                    )
                )

    df_ejemplo = pd.DataFrame(registros)
    totales_mes = df_ejemplo.groupby("fecha")["exposicion"].transform("sum")
    df_ejemplo["pct_cartera_mensual"] = (df_ejemplo["exposicion"] / totales_mes).round(6)
    df_ejemplo = df_ejemplo[COLUMNAS_REQUERIDAS].sort_values(["fecha", "sector", "region"]).reset_index(drop=True)
    return df_ejemplo


@st.cache_data(show_spinner=False)
def _datos_ejemplo_csv_texto() -> str:
    buffer = io.StringIO()
    _generar_datos_ejemplo().to_csv(buffer, index=False)
    return buffer.getvalue()


# =============================================================================
# SECCIÓN 6 — Aplicación Streamlit
# =============================================================================
st.set_page_config(
    page_title="Optimizador de Cartera de Arrendamiento — Frontera Eficiente",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(css_global(), unsafe_allow_html=True)


# --- Funciones cacheadas ---
@st.cache_data(show_spinner="Cargando y validando datos...")
def _cargar_datos_cache(bytes_datos: bytes | None):
    if bytes_datos is not None:
        resultado = cargar_datos(io.BytesIO(bytes_datos))
    else:
        resultado = cargar_datos(io.StringIO(_datos_ejemplo_csv_texto()))
    return resultado.df, resultado.advertencias


@st.cache_data(show_spinner=False)
def _estadisticos_cache(matriz_json: str, anualizar: bool):
    matriz = pd.read_json(io.StringIO(matriz_json), orient="split")
    matriz.index = pd.to_datetime(matriz.index)
    return calcular_estadisticos(matriz, anualizar=anualizar)


@st.cache_data(show_spinner=False)
def _gmv_cache(mu_tuple, cov_tuple, min_tuple, max_tuple):
    """Portafolio de mínima varianza global (sin retorno objetivo), usado para acotar el slider."""
    mu = np.array(mu_tuple)
    cov = np.array(cov_tuple).reshape(len(mu), len(mu))
    lim_min = np.array(min_tuple)
    lim_max = np.array(max_tuple)
    return optimizar_min_varianza(mu, cov, lim_min, lim_max, retorno_objetivo=None)


@st.cache_data(show_spinner="Calculando frontera eficiente...")
def _frontera_cache(mu_tuple, cov_tuple, min_tuple, max_tuple, n_puntos):
    mu = np.array(mu_tuple)
    cov = np.array(cov_tuple).reshape(len(mu), len(mu))
    lim_min = np.array(min_tuple)
    lim_max = np.array(max_tuple)
    return calcular_frontera_eficiente(mu, cov, lim_min, lim_max, n_puntos=n_puntos)


@st.cache_data(show_spinner="Simulando portafolios aleatorios...")
def _montecarlo_cache(mu_tuple, cov_tuple, min_tuple, max_tuple, n_sim, semilla):
    mu = np.array(mu_tuple)
    cov = np.array(cov_tuple).reshape(len(mu), len(mu))
    lim_min = np.array(min_tuple)
    lim_max = np.array(max_tuple)
    return simulacion_portafolios_aleatorios(mu, cov, lim_min, lim_max, n_sim=n_sim, semilla=semilla)


def _fmt_pct(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "s/d"
    return f"{x * 100:,.{dec}f}%"


def _fmt_num(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "s/d"
    return f"{x:,.{dec}f}"


# --- Encabezado ---
st.markdown(
    """
    <div class="app-header">
        <h1>📈 Optimizador de Cartera de Arrendamiento Financiero</h1>
        <p>Frontera Eficiente (Markowitz) aplicada a segmentos Sector × Región · Panel de asignación óptima de capital</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: carga de datos ---
st.sidebar.markdown("## ⚙️ Parámetros del modelo")

st.sidebar.markdown("#### 1. Fuente de datos")
archivo_subido = st.sidebar.file_uploader(
    "Cargar archivo CSV de cartera",
    type=["csv"],
    help=(
        "Debe respetar la estructura de columnas: fecha, sector, region, segmento, "
        "exposicion, num_contratos, tasa_bruta_mensual, pd_mensual, lgd, "
        "perdida_esperada_monto, perdida_esperada_tasa, rendimiento_neto_mensual, "
        "pct_cartera_mensual, limite_maximo_ejemplo_pct."
    ),
)

usando_ejemplo = archivo_subido is None
bytes_datos = archivo_subido.getvalue() if archivo_subido is not None else None

try:
    df, advertencias = _cargar_datos_cache(bytes_datos)
except ErrorEstructuraDatos as e:
    st.error(f"❌ {e}")
    st.stop()

if usando_ejemplo:
    st.sidebar.info("📄 Usando datos de ejemplo sintéticos (`base_datos_cartera_sintetica.csv`).")
else:
    st.sidebar.success(f"✅ Archivo cargado: {archivo_subido.name}")

if advertencias:
    with st.sidebar.expander(f"⚠️ Advertencias de carga ({len(advertencias)})"):
        for a in advertencias:
            st.write(f"- {a}")

# --- Sidebar: rango de fechas y segmentos ---
st.sidebar.markdown("#### 2. Periodo de análisis")
fecha_min, fecha_max = rango_fechas(df)
rango_sel = st.sidebar.date_input(
    "Rango de fechas (histórico a considerar)",
    value=(fecha_min.date(), fecha_max.date()),
    min_value=fecha_min.date(),
    max_value=fecha_max.date(),
)
if isinstance(rango_sel, tuple) and len(rango_sel) == 2:
    fecha_ini, fecha_fin = rango_sel
else:
    fecha_ini, fecha_fin = fecha_min.date(), fecha_max.date()

st.sidebar.markdown("#### 3. Segmentos a incluir")
todos_segmentos = lista_segmentos(df)
segmentos_sel = st.sidebar.multiselect(
    "Segmentos (sector × región)",
    options=todos_segmentos,
    default=todos_segmentos,
)

if len(segmentos_sel) < 2:
    st.warning(
        "⚠️ Seleccione al menos **2 segmentos** para poder estimar una matriz de "
        "covarianza y construir la frontera eficiente."
    )
    st.stop()

df_filtrado = filtrar(df, fecha_ini, fecha_fin, segmentos_sel)
if df_filtrado.empty:
    st.error("❌ No hay datos disponibles para la combinación de fecha/segmentos seleccionada.")
    st.stop()

matriz_rend = matriz_rendimientos(df_filtrado)
n_meses_validos = matriz_rend.dropna(how="any").shape[0]
if n_meses_validos < 3:
    st.warning(
        f"⚠️ Solo hay {n_meses_validos} periodo(s) mensuales completos para los segmentos "
        "seleccionados. Las estimaciones de riesgo/correlación pueden ser poco confiables "
        "con historia tan corta."
    )

# --- Sidebar: límites de concentración por segmento ---
st.sidebar.markdown("#### 4. Límites de concentración por segmento")

limites_ejemplo = limites_maximos_ejemplo(df_filtrado, segmentos_sel)
usar_min_personalizado = st.sidebar.checkbox(
    "Aplicar límite mínimo por segmento (default: 0%)", value=False
)

tabla_limites_base = pd.DataFrame(
    {
        "segmento": segmentos_sel,
        "limite_min_%": 0.0,
        "limite_max_%": (limites_ejemplo.reindex(segmentos_sel).fillna(0.2) * 100).round(2),
    }
)

with st.sidebar.expander("Editar límites por segmento", expanded=False):
    st.caption(
        "Precargados desde `limite_maximo_ejemplo_pct` del archivo. Sustituya por límites "
        "reales de política interna o regulatorios cuando use una base real."
    )
    tabla_limites_editada = st.data_editor(
        tabla_limites_base,
        column_config={
            "segmento": st.column_config.TextColumn("Segmento", disabled=True),
            "limite_min_%": st.column_config.NumberColumn(
                "Mín. %", min_value=0.0, max_value=100.0, step=0.5, disabled=not usar_min_personalizado
            ),
            "limite_max_%": st.column_config.NumberColumn(
                "Máx. %", min_value=0.0, max_value=100.0, step=0.5
            ),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="editor_limites",
    )

limites_min_pct = (
    tabla_limites_editada.set_index("segmento")["limite_min_%"]
    if usar_min_personalizado
    else pd.Series(0.0, index=segmentos_sel)
)
limites_max_pct = tabla_limites_editada.set_index("segmento")["limite_max_%"]

limites_min = (limites_min_pct.reindex(segmentos_sel).fillna(0.0) / 100).to_numpy()
limites_max = (limites_max_pct.reindex(segmentos_sel).fillna(20.0) / 100).to_numpy()
limites_max = np.maximum(limites_max, limites_min + 1e-6)

if limites_max.sum() < 1.0:
    st.sidebar.error(
        f"⚠️ La suma de límites máximos ({limites_max.sum() * 100:.1f}%) es menor a 100%: "
        "no es posible asignar toda la cartera. Aumente algún límite máximo."
    )
    st.stop()
if limites_min.sum() > 1.0:
    st.sidebar.error(
        f"⚠️ La suma de límites mínimos ({limites_min.sum() * 100:.1f}%) supera 100%: "
        "el problema no tiene solución factible. Reduzca algún límite mínimo."
    )
    st.stop()

# --- Sidebar: configuración de cálculo ---
st.sidebar.markdown("#### 5. Métricas y referencia")
anualizar = st.sidebar.checkbox("Anualizar retorno y riesgo (× 12 / × √12)", value=True)
costo_fondeo_mensual_pct = st.sidebar.number_input(
    "Costo de fondeo mensual de referencia (%)",
    min_value=0.0,
    max_value=10.0,
    value=0.90,
    step=0.05,
    help="Se usa como r_referencia en el ratio retorno/riesgo tipo Sharpe adaptado a crédito.",
)
r_referencia = (costo_fondeo_mensual_pct / 100) * (12 if anualizar else 1)

# --- Cálculo de estadísticos (mu, sigma, covarianza, correlación) ---
# (Se calcula aquí, antes de las secciones 6 y 7, porque el slider de
# "retorno objetivo" de la sección 6 necesita conocer el rango factible de
# retornos ANTES de dibujarse — de lo contrario ese control terminaba
# apareciendo, en el orden real de renderizado, después de los controles de
# la sección 7 en vez de justo debajo de su propio encabezado.)
matriz_json = (
    matriz_rend.reset_index()
    .assign(fecha=lambda d: d["fecha"].astype(str))
    .set_index("fecha")
    .to_json(orient="split")
)
stats = _estadisticos_cache(matriz_json, anualizar)

mu = stats.mu.reindex(segmentos_sel)
cov = stats.cov.reindex(index=segmentos_sel, columns=segmentos_sel)
corr = stats.corr.reindex(index=segmentos_sel, columns=segmentos_sel)

mu_arr = mu.to_numpy()
cov_arr = cov.to_numpy()

r_min_fact, r_max_fact = rango_retorno_factible(mu_arr, limites_min, limites_max)
if r_min_fact is None:
    st.sidebar.error("⚠️ No se encontró una combinación de pesos factible con los límites actuales.")
    st.stop()

# Portafolio de mínima varianza global (GMV): retorno más bajo que en verdad
# pertenece a la frontera EFICIENTE (la rama inferior, con retorno menor al
# del GMV, queda dominada y no debe poder seleccionarse como "óptima").
res_gmv_sidebar = _gmv_cache(tuple(mu_arr), tuple(cov_arr.flatten()), tuple(limites_min), tuple(limites_max))
r_gmv_sidebar = res_gmv_sidebar.retorno if res_gmv_sidebar.exito else r_min_fact

# --- Sidebar: punto sobre la frontera eficiente ---
st.sidebar.markdown("#### 6. Punto sobre la frontera eficiente")
modo_objetivo = st.sidebar.radio(
    "Definir el portafolio óptimo por:",
    options=["Retorno objetivo", "Aversión al riesgo (λ)"],
    index=0,
)

retorno_objetivo = None
aversion_riesgo = None

if modo_objetivo == "Retorno objetivo":
    unidad = "anual" if anualizar else "mensual"
    retorno_objetivo_pct = st.sidebar.slider(
        f"Retorno objetivo {unidad} (%)",
        min_value=float(r_gmv_sidebar * 100),
        max_value=float(r_max_fact * 100),
        value=float(np.clip(mu_arr.mean() * 100, r_gmv_sidebar * 100, r_max_fact * 100)),
        step=0.01,
        format="%.2f",
        help=(
            "El mínimo posible es el retorno del portafolio de mínima varianza global: "
            "por debajo de ese punto ya no se está sobre la frontera eficiente."
        ),
    )
    retorno_objetivo = retorno_objetivo_pct / 100
else:
    aversion_riesgo = st.sidebar.slider(
        "Nivel de aversión al riesgo (λ) — mayor = más conservador",
        min_value=0.5,
        max_value=50.0,
        value=5.0,
        step=0.5,
    )

# --- Sidebar: método de optimización ---
st.sidebar.markdown("#### 7. Método de optimización")
metodo_opt = st.sidebar.radio(
    "Método",
    options=["Optimización analítica (mínima varianza)", "Simulación de portafolios aleatorios (Monte Carlo)"],
    index=0,
)
n_puntos_frontera = st.sidebar.slider("Puntos en la frontera eficiente", 10, 80, 40, step=5)
n_sim_mc = 3000
if metodo_opt.startswith("Simulación"):
    n_sim_mc = st.sidebar.slider("Número de portafolios simulados", 500, 10000, 3000, step=500)

# --- Distribución actual y exposición ---
dist_actual = distribucion_actual(df_filtrado, segmentos_sel)
w_actual = dist_actual.to_numpy()

r_actual = rendimiento_portafolio(w_actual, mu_arr)
sigma_actual = riesgo_portafolio(w_actual, cov_arr)
ratio_actual = ratio_retorno_riesgo(r_actual, sigma_actual, r_referencia)
hhi_actual = hhi(w_actual)

# --- Frontera eficiente (siempre calculada) y Monte Carlo (si aplica) ---
frontera = _frontera_cache(
    tuple(mu_arr), tuple(cov_arr.flatten()), tuple(limites_min), tuple(limites_max), n_puntos_frontera
)

nube_mc = None
if metodo_opt.startswith("Simulación"):
    nube_mc = _montecarlo_cache(
        tuple(mu_arr), tuple(cov_arr.flatten()), tuple(limites_min), tuple(limites_max), n_sim_mc, 42
    )

# --- Portafolio óptimo según método y modo seleccionados ---
mensaje_opt = ""
w_optimo = None

if metodo_opt.startswith("Optimización analítica"):
    if modo_objetivo == "Retorno objetivo":
        res = optimizar_min_varianza(mu_arr, cov_arr, limites_min, limites_max, retorno_objetivo)
    else:
        res = optimizar_max_utilidad(mu_arr, cov_arr, limites_min, limites_max, aversion_riesgo)
    if res.exito:
        w_optimo = res.pesos
    else:
        mensaje_opt = f"La optimización no convergió: {res.mensaje}"
else:
    if nube_mc is None:
        mensaje_opt = (
            "No fue posible generar portafolios aleatorios que respeten los límites "
            "definidos. Intente relajar los límites de concentración."
        )
    else:
        pesos_mc, ret_mc, riesgo_mc = nube_mc
        if modo_objetivo == "Retorno objetivo":
            cumplen = ret_mc >= retorno_objetivo
            if cumplen.any():
                idx = np.argmin(np.where(cumplen, riesgo_mc, np.inf))
            else:
                idx = int(np.argmin(np.abs(ret_mc - retorno_objetivo)))
            w_optimo = pesos_mc[idx]
        else:
            utilidad = ret_mc - 0.5 * aversion_riesgo * (riesgo_mc ** 2)
            idx = int(np.argmax(utilidad))
            w_optimo = pesos_mc[idx]

if mensaje_opt:
    st.warning(f"⚠️ {mensaje_opt}")

if w_optimo is not None:
    r_optimo = rendimiento_portafolio(w_optimo, mu_arr)
    sigma_optimo = riesgo_portafolio(w_optimo, cov_arr)
    ratio_optimo = ratio_retorno_riesgo(r_optimo, sigma_optimo, r_referencia)
    hhi_optimo = hhi(w_optimo)
else:
    r_optimo = sigma_optimo = ratio_optimo = hhi_optimo = None

# --- Resumen de datos utilizados ---
resumen = resumen_calidad(df_filtrado)
st.caption(
    f"Periodo analizado: **{resumen['fecha_min'].strftime('%b %Y')} – {resumen['fecha_max'].strftime('%b %Y')}** · "
    f"**{len(segmentos_sel)}** segmentos ({resumen['n_sectores']} sectores × {resumen['n_regiones']} regiones) · "
    f"**{n_meses_validos}** periodos mensuales completos · Métricas en base "
    f"**{'anualizada' if anualizar else 'mensual'}**"
)

# --- KPIs ---
# NOTA: cada bloque ("Cartera actual" / "Cartera óptima") ocupa el ancho
# completo en su propia fila de 4 columnas. Antes se anidaban 2 columnas de
# 50% con 4 sub-columnas cada una (8 tarjetas angostas en una sola fila), lo
# que truncaba los valores (ej. "17.1…") por falta de espacio horizontal.
st.markdown('<div class="section-title">Métricas resumen</div>', unsafe_allow_html=True)

st.markdown("**📌 Cartera actual**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Retorno esperado", _fmt_pct(r_actual))
c2.metric("Volatilidad", _fmt_pct(sigma_actual))
c3.metric("Ratio retorno/riesgo", _fmt_num(ratio_actual))
c4.metric("HHI concentración", _fmt_num(hhi_actual, 4))

st.markdown("**⭐ Cartera óptima seleccionada**")
if w_optimo is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno esperado", _fmt_pct(r_optimo), delta=_fmt_pct(r_optimo - r_actual))
    c2.metric("Volatilidad", _fmt_pct(sigma_optimo), delta=_fmt_pct(sigma_optimo - sigma_actual), delta_color="inverse")
    c3.metric("Ratio retorno/riesgo", _fmt_num(ratio_optimo), delta=_fmt_num(ratio_optimo - ratio_actual))
    c4.metric("HHI concentración", _fmt_num(hhi_optimo, 4), delta=_fmt_num(hhi_optimo - hhi_actual, 4), delta_color="inverse")
else:
    st.info("No disponible.")

# --- Frontera eficiente ---
st.markdown('<div class="section-title">Frontera eficiente</div>', unsafe_allow_html=True)

fig_frontera = go.Figure()

if nube_mc is not None:
    pesos_mc, ret_mc, riesgo_mc = nube_mc
    ratio_mc = (ret_mc - r_referencia) / np.where(riesgo_mc > 0, riesgo_mc, np.nan)
    fig_frontera.add_trace(
        go.Scattergl(
            x=riesgo_mc,
            y=ret_mc,
            mode="markers",
            name="Portafolios simulados",
            marker=dict(
                size=5,
                color=ratio_mc,
                colorscale=[[0, COLOR_GRAY_LIGHT], [1, COLOR_PRIMARY]],
                showscale=False,
                opacity=0.45,
            ),
            hovertemplate="Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
        )
    )

if frontera is not None:
    fig_frontera.add_trace(
        go.Scatter(
            x=frontera.riesgos,
            y=frontera.retornos,
            mode="lines",
            name="Frontera eficiente",
            line=dict(color=COLOR_PRIMARY_DARK, width=3),
            hovertemplate="Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
        )
    )
else:
    st.info("No fue posible calcular la frontera eficiente con los límites actuales.")

fig_frontera.add_trace(
    go.Scatter(
        x=[sigma_actual],
        y=[r_actual],
        mode="markers",
        name="Cartera actual",
        marker=dict(color=COLOR_GRAY, size=15, symbol="diamond", line=dict(width=1.5, color="white")),
        hovertemplate="Cartera actual<br>Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
    )
)

if w_optimo is not None:
    fig_frontera.add_trace(
        go.Scatter(
            x=[sigma_optimo],
            y=[r_optimo],
            mode="markers",
            name="Cartera óptima seleccionada",
            marker=dict(color=COLOR_POSITIVE, size=17, symbol="star", line=dict(width=1.5, color="white")),
            hovertemplate="Cartera óptima<br>Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
        )
    )

fig_frontera.update_layout(
    xaxis_title="Riesgo (volatilidad, σ)",
    yaxis_title="Retorno esperado (μ)",
    xaxis_tickformat=".1%",
    yaxis_tickformat=".1%",
)
layout_plotly(fig_frontera, altura=460)
st.plotly_chart(fig_frontera, use_container_width=True)

# --- Tabla de pesos + gráfica comparativa ---
st.markdown('<div class="section-title">Asignación óptima vs. cartera actual</div>', unsafe_allow_html=True)

tabla_pesos = None
if w_optimo is not None:
    tabla_pesos = pd.DataFrame(
        {
            "segmento": segmentos_sel,
            "peso_actual_%": (w_actual * 100).round(2),
            "peso_optimo_%": (w_optimo * 100).round(2),
        }
    )
    tabla_pesos["diferencia_pp"] = (tabla_pesos["peso_optimo_%"] - tabla_pesos["peso_actual_%"]).round(2)
    tabla_pesos["limite_max_%"] = (limites_max * 100).round(2)
    tabla_pesos = tabla_pesos.sort_values("peso_optimo_%", ascending=False).reset_index(drop=True)

    col_izq, col_der = st.columns([1, 1.2])

    with col_izq:
        st.dataframe(
            tabla_pesos,
            use_container_width=True,
            hide_index=True,
            column_config={
                "segmento": "Segmento",
                "peso_actual_%": st.column_config.NumberColumn("Peso actual (%)", format="%.2f"),
                "peso_optimo_%": st.column_config.NumberColumn("Peso óptimo (%)", format="%.2f"),
                "diferencia_pp": st.column_config.NumberColumn("Δ (p.p.)", format="%.2f"),
                "limite_max_%": st.column_config.NumberColumn("Límite máx. (%)", format="%.2f"),
            },
            height=420,
        )

    with col_der:
        tabla_grafica = tabla_pesos.sort_values("peso_optimo_%", ascending=True)
        fig_barras = go.Figure()
        fig_barras.add_trace(
            go.Bar(
                y=tabla_grafica["segmento"],
                x=tabla_grafica["peso_actual_%"],
                name="Actual",
                orientation="h",
                marker_color=COLOR_GRAY,
            )
        )
        fig_barras.add_trace(
            go.Bar(
                y=tabla_grafica["segmento"],
                x=tabla_grafica["peso_optimo_%"],
                name="Óptimo",
                orientation="h",
                marker_color=COLOR_PRIMARY,
            )
        )
        fig_barras.update_layout(barmode="group", xaxis_title="Peso en cartera (%)", height=420)
        layout_plotly(fig_barras, altura=420)
        st.plotly_chart(fig_barras, use_container_width=True)
else:
    st.info("No hay una asignación óptima disponible para mostrar.")

# --- Heatmap de correlación ---
st.markdown('<div class="section-title">Matriz de correlación entre segmentos</div>', unsafe_allow_html=True)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=corr.to_numpy(),
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=ESCALA_CORRELACION,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="ρ"),
        hovertemplate="%{y} × %{x}<br>Correlación: %{z:.2f}<extra></extra>",
    )
)
fig_heatmap.update_layout(height=max(420, 22 * len(segmentos_sel)))
layout_plotly(fig_heatmap, leyenda=False)
st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Descarga de resultados ---
st.markdown('<div class="section-title">Descargar resultados</div>', unsafe_allow_html=True)

if w_optimo is not None:
    tabla_metricas = pd.DataFrame(
        {
            "métrica": ["Retorno esperado", "Volatilidad", "Ratio retorno/riesgo", "HHI concentración"],
            "cartera_actual": [r_actual, sigma_actual, ratio_actual, hhi_actual],
            "cartera_optima": [r_optimo, sigma_optimo, ratio_optimo, hhi_optimo],
        }
    )

    col_csv, col_xlsx = st.columns(2)

    csv_bytes = tabla_pesos.to_csv(index=False).encode("utf-8-sig")
    col_csv.download_button(
        "⬇️ Descargar pesos óptimos (CSV)",
        data=csv_bytes,
        file_name=f"pesos_optimos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    buffer_excel = io.BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
        tabla_pesos.to_excel(writer, sheet_name="pesos_optimos", index=False)
        tabla_metricas.to_excel(writer, sheet_name="metricas_resumen", index=False)
        corr.to_excel(writer, sheet_name="matriz_correlacion")
    col_xlsx.download_button(
        "⬇️ Descargar resultados completos (Excel)",
        data=buffer_excel.getvalue(),
        file_name=f"resultados_optimizacion_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# --- Datos filtrados (detalle) ---
with st.expander("🔎 Ver datos filtrados utilizados en el cálculo"):
    st.dataframe(df_filtrado, use_container_width=True, hide_index=True)

st.markdown(
    f"""
    <div style="text-align:center; color:{COLOR_GRAY}; font-size:0.78rem; margin-top:24px;">
        Optimizador de Cartera — Teoría Moderna de Portafolios (Markowitz) ·
        Herramienta de apoyo a la decisión, no constituye recomendación de inversión.
    </div>
    """,
    unsafe_allow_html=True,
)
