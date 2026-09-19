"""
Generador de base de datos sintética de cartera de arrendamiento financiero.

Este script se incluye únicamente como herramienta de apoyo para crear el
archivo de ejemplo `base_datos_cartera_sintetica.csv`. NO es parte del
pipeline de la aplicación Streamlit: la app lee el CSV ya generado (o
cualquier archivo real que respete la misma estructura de columnas).

Uso:
    python generar_datos_sinteticos.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
SEMILLA = 42
rng = np.random.default_rng(SEMILLA)

SECTORES = [
    "Manufactura",
    "Comercio",
    "Servicios",
    "Transporte",
    "Construccion",
    "Agroindustria",
]

REGIONES = ["Norte", "Centro", "Bajio", "Sur"]

# 48 meses de historia (4 años), terminando en dic-2024
FECHAS = pd.date_range(end="2024-12-31", periods=48, freq="ME")

# Parámetros base por sector (nivel de riesgo/retorno "típico" del sector)
# tasa_bruta_mensual y pd_mensual en proporción (no en %), ej. 0.018 = 1.8%
PARAMS_SECTOR = {
    "Manufactura":    dict(tasa_base=0.0165, pd_base=0.0060, lgd_base=0.40, tam_base=180_000_000),
    "Comercio":       dict(tasa_base=0.0180, pd_base=0.0090, lgd_base=0.45, tam_base=150_000_000),
    "Servicios":      dict(tasa_base=0.0155, pd_base=0.0050, lgd_base=0.38, tam_base=120_000_000),
    "Transporte":     dict(tasa_base=0.0195, pd_base=0.0110, lgd_base=0.50, tam_base=110_000_000),
    "Construccion":   dict(tasa_base=0.0210, pd_base=0.0130, lgd_base=0.55, tam_base=95_000_000),
    "Agroindustria":  dict(tasa_base=0.0185, pd_base=0.0100, lgd_base=0.48, tam_base=80_000_000),
}

# Ajustes multiplicativos por región (algunas regiones más/menos volátiles)
AJUSTE_REGION = {
    "Norte":  dict(tasa_mult=1.05, pd_mult=0.95, tam_mult=1.15),
    "Centro": dict(tasa_mult=1.00, pd_mult=1.00, tam_mult=1.30),
    "Bajio":  dict(tasa_mult=0.98, pd_mult=0.90, tam_mult=1.00),
    "Sur":    dict(tasa_mult=1.10, pd_mult=1.20, tam_mult=0.70),
}

# Límite máximo "de ejemplo" por segmento (plantilla, NO es un límite real)
# En una base real este valor se sustituye por políticas internas/regulatorias.
LIMITE_MIN, LIMITE_MAX = 0.08, 0.22

registros = []

for sector in SECTORES:
    ps = PARAMS_SECTOR[sector]
    for region in REGIONES:
        ar = AJUSTE_REGION[region]
        segmento = f"{sector} - {region}"

        tasa_media = ps["tasa_base"] * ar["tasa_mult"]
        pd_media = ps["pd_base"] * ar["pd_mult"]
        lgd_media = ps["lgd_base"]
        tam_medio = ps["tam_base"] * ar["tam_mult"]

        # límite de ejemplo, fijo por segmento a lo largo del tiempo
        limite_pct = rng.uniform(LIMITE_MIN, LIMITE_MAX)

        # --- simulación de series mensuales con autocorrelación simple (AR1) ---
        n = len(FECHAS)

        # tasa bruta mensual: pequeña variación alrededor de la media del segmento
        ruido_tasa = rng.normal(0, tasa_media * 0.08, n)
        tasa_bruta = np.clip(tasa_media + np.cumsum(ruido_tasa) * 0.15 / np.sqrt(n)
                              + rng.normal(0, tasa_media * 0.05, n), 0.004, 0.05)

        # PD mensual: proceso AR(1) alrededor de la media del segmento, truncado
        pd_serie = np.zeros(n)
        pd_serie[0] = pd_media
        for t in range(1, n):
            choque = rng.normal(0, pd_media * 0.25)
            pd_serie[t] = 0.85 * pd_serie[t - 1] + 0.15 * pd_media + choque
        pd_serie = np.clip(pd_serie, 0.0005, 0.08)

        # LGD: variación leve
        lgd_serie = np.clip(lgd_media + rng.normal(0, 0.03, n), 0.15, 0.85)

        # Exposición: tendencia + estacionalidad leve + ruido (nivel de cartera)
        tendencia = np.linspace(0, rng.uniform(-0.10, 0.35), n)
        estacional = 0.03 * np.sin(np.linspace(0, 4 * np.pi, n))
        ruido_exp = rng.normal(0, 0.04, n)
        exposicion = tam_medio * (1 + tendencia + estacional + np.cumsum(ruido_exp) * 0.05)
        exposicion = np.clip(exposicion, tam_medio * 0.3, None)

        # Tamaño promedio de contrato por sector (aprox.), con algo de variación
        tam_contrato_prom = {
            "Manufactura": 2_800_000, "Comercio": 1_200_000, "Servicios": 900_000,
            "Transporte": 1_600_000, "Construccion": 3_500_000, "Agroindustria": 2_200_000,
        }[sector]
        num_contratos = np.maximum(
            5, np.round(exposicion / tam_contrato_prom * rng.uniform(0.85, 1.15, n))
        ).astype(int)

        perdida_esperada_tasa = pd_serie * lgd_serie
        perdida_esperada_monto = perdida_esperada_tasa * exposicion
        rendimiento_neto_mensual = tasa_bruta - perdida_esperada_tasa

        for i, fecha in enumerate(FECHAS):
            registros.append(dict(
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
            ))

df = pd.DataFrame(registros)

# pct_cartera_mensual: participación de cada segmento en la exposición total de ese mes
totales_mes = df.groupby("fecha")["exposicion"].transform("sum")
df["pct_cartera_mensual"] = (df["exposicion"] / totales_mes).round(6)

# Orden final de columnas según especificación
COLUMNAS = [
    "fecha", "sector", "region", "segmento", "exposicion", "num_contratos",
    "tasa_bruta_mensual", "pd_mensual", "lgd", "perdida_esperada_monto",
    "perdida_esperada_tasa", "rendimiento_neto_mensual", "pct_cartera_mensual",
    "limite_maximo_ejemplo_pct",
]
df = df[COLUMNAS].sort_values(["fecha", "sector", "region"]).reset_index(drop=True)

df.to_csv("base_datos_cartera_sintetica.csv", index=False, encoding="utf-8-sig")
print(f"Archivo generado: base_datos_cartera_sintetica.csv ({len(df)} filas, "
      f"{df['segmento'].nunique()} segmentos, {df['fecha'].nunique()} meses)")
