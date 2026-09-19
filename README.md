# Optimizador de Cartera de Arrendamiento Financiero — Frontera Eficiente (Markowitz)

Aplicación en Streamlit que aplica la **Teoría Moderna de Portafolios (Markowitz)**
a la cartera de una **Arrendadora Financiera**, tratando cada combinación
**sector económico × región geográfica** como un "activo" del modelo.

A partir del histórico mensual de `rendimiento_neto_mensual` por segmento, la
app estima retorno esperado y riesgo (volatilidad), construye la matriz de
covarianza/correlación, traza la **frontera eficiente**, y calcula la
**asignación óptima de capital** sujeta a límites de concentración por
segmento — comparándola contra la distribución actual de la cartera.

> ⚠️ Esta herramienta es de apoyo a la decisión y no constituye, por sí sola,
> una recomendación de inversión o de política de originación.

---

## 1. Estructura del repositorio

```
cartera-optimizacion/
├── app.py                          # Aplicación completa (datos + modelo Markowitz + interfaz)
├── requirements.txt                # Dependencias de Python
├── README.md                       # Este archivo
├── .streamlit/
│   └── config.toml                 # Tema visual (colores, tipografía base)
└── data/
    ├── base_datos_cartera_sintetica.csv   # Base de datos de ejemplo (sintética)
    └── generar_datos_sinteticos.py        # Script para regenerar el ejemplo (opcional)
```

> **Nota de diseño:** toda la lógica (esquema de columnas, carga/validación de
> datos, modelo de optimización Markowitz y estilos) vive dentro de un único
> `app.py`, en vez de repartirse en un paquete `src/` con varios módulos.
> Esto es deliberado: al subir un repositorio manualmente por la interfaz web
> de GitHub (arrastrando archivos en vez de usar `git push`), es muy común
> que una subcarpeta no se suba completa y la app truene en Streamlit Cloud
> con `ModuleNotFoundError: No module named 'src'`. Con un solo archivo ese
> riesgo desaparece por completo.

---

## 2. Estructura de datos esperada

La app funciona con **cualquier archivo CSV** que respete exactamente estos
nombres de columna y su significado (para poder sustituir el archivo de
ejemplo por una base de datos real sin modificar el código):

| Columna | Descripción |
|---|---|
| `fecha` | Fecha de corte mensual del registro (fin de mes). |
| `sector` | Sector económico del segmento. |
| `region` | Región geográfica del segmento. |
| `segmento` | Identificador del segmento (combinación sector × región). |
| `exposicion` | Saldo/exposición vigente del segmento en el mes. |
| `num_contratos` | Número de contratos de arrendamiento activos. |
| `tasa_bruta_mensual` | Tasa activa bruta ponderada mensual (proporción, ej. `0.018` = 1.8%). |
| `pd_mensual` | Probabilidad de incumplimiento mensual ponderada. |
| `lgd` | Loss Given Default (severidad de la pérdida) ponderada. |
| `perdida_esperada_monto` | `pd_mensual × lgd × exposicion`. |
| `perdida_esperada_tasa` | `pd_mensual × lgd`. |
| `rendimiento_neto_mensual` | Tasa bruta ponderada **menos** pérdida esperada ponderada. **Ya viene calculada — la app no vuelve a restar la pérdida esperada.** |
| `pct_cartera_mensual` | Participación (%) del segmento sobre la cartera total en ese mes. |
| `limite_maximo_ejemplo_pct` | Límite máximo de concentración de **ejemplo/plantilla**. En producción debe sustituirse por límites reales de política interna o regulatorios (editable desde la interfaz). |

El archivo `data/base_datos_cartera_sintetica.csv` es un ejemplo generado
sintéticamente (24 segmentos = 6 sectores × 4 regiones, 48 meses de historia)
únicamente para fines de prueba y demostración.

**Para usar datos reales:** simplemente cargue su propio archivo CSV desde la
barra lateral de la app ("Cargar archivo CSV de cartera"); no es necesario
modificar ningún archivo del repositorio.

---

## 3. Modelo de optimización

Por segmento, a partir del histórico de `rendimiento_neto_mensual`:

- **Retorno esperado:** `μᵢ = promedio histórico mensual` (opcionalmente anualizado ×12)
- **Riesgo:** `σᵢ = desviación estándar histórica mensual` (opcionalmente anualizado ×√12)
- **Matriz de covarianza `Σ`** entre segmentos, sobre la misma serie histórica

A nivel portafolio, dado un vector de pesos `w`:

- Retorno esperado: `Rₚ = wᵀ μ`
- Varianza: `σₚ² = wᵀ Σ w`; volatilidad `σₚ = √σₚ²`

**Restricciones:** `Σwᵢ = 1`, `0 ≤ wᵢ ≤ límite_máximo_segmento` (y opcionalmente un mínimo).

**Función objetivo** (frontera punto por punto): para cada retorno objetivo `R*`,
minimizar `σₚ²` sujeto a `Rₚ = R*` y las restricciones anteriores
(`scipy.optimize.minimize`, método SLSQP). El rango factible de `R*` se
calcula primero vía programación lineal (`scipy.optimize.linprog`).

**Métricas adicionales:**

- Ratio retorno/riesgo tipo Sharpe adaptado a crédito: `(Rₚ − r_referencia) / σₚ`,
  usando como `r_referencia` el **costo de fondeo** de la arrendadora (editable
  en la interfaz) en vez de una tasa libre de riesgo tradicional.
- Índice de concentración Herfindahl-Hirschman: `HHI = Σwᵢ²`.

También se incluye un **método alternativo de simulación Monte Carlo**
(muestreo de portafolios aleatorios factibles vía distribución Dirichlet) como
forma adicional de explorar el espacio de portafolios y seleccionar el punto
óptimo según el criterio elegido (retorno objetivo o aversión al riesgo).

---

## 4. Instalación y ejecución local

Requiere Python 3.10+.

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/<tu-repositorio>.git
cd cartera-optimizacion

# 2. Crear un entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La app abrirá automáticamente en `http://localhost:8501`. Si no carga ningún
archivo, usará por defecto datos sintéticos de ejemplo equivalentes a los de
`data/base_datos_cartera_sintetica.csv` — generados **en memoria** por la
propia app (misma lógica y semilla que `data/generar_datos_sinteticos.py`),
por lo que la demo funciona incluso si esa carpeta no está presente.

---

## 5. Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público o privado).
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión
   con tu cuenta de GitHub.
3. Clic en **"New app"** → selecciona el repositorio, la rama (`main`) y el
   archivo principal: `app.py`.
4. Clic en **"Deploy"**. Streamlit instalará automáticamente las dependencias
   de `requirements.txt` y publicará la app en una URL pública tipo
   `https://<nombre-app>.streamlit.app`.
5. Cada `git push` a la rama configurada actualiza automáticamente la app
   desplegada.

No se requieren credenciales ni variables de entorno adicionales para
ejecutar la app con datos de ejemplo. Si en el futuro se conecta a una base
de datos real, agregue las credenciales correspondientes en
**Settings → Secrets** del panel de Streamlit Cloud (archivo `secrets.toml`,
ya excluido de este repositorio vía `.gitignore`).

---

## 6. Regenerar la base de datos de ejemplo (opcional)

```bash
cd data
python generar_datos_sinteticos.py
```

Esto sobrescribe `base_datos_cartera_sintetica.csv` con una nueva simulación
(semilla fija = 42, reproducible).

---

## 7. Solución de problemas al desplegar

**Error `ModuleNotFoundError: No module named 'src'` (o similar) en Streamlit Cloud**

Este error ocurre cuando el repositorio en GitHub no contiene todos los
archivos/carpetas del proyecto — típicamente porque se subieron arrastrando
archivos sueltos desde la interfaz web de GitHub en vez de subir la carpeta
completa. La versión actual de este proyecto ya no depende de ninguna
subcarpeta de código (todo vive en `app.py`), por lo que este error
específico no debería volver a ocurrir. Si aun así ves un error de
`ModuleNotFoundError` con otro nombre de módulo, verifica:

1. Que `requirements.txt` esté en la **raíz** del repositorio (mismo nivel
   que `app.py`), no dentro de una subcarpeta.
2. Que, al crear la app en Streamlit Cloud, el campo "Main file path" apunte
   exactamente a `app.py`.
3. En "Manage app" → "Reboot app" después de cualquier cambio al
   `requirements.txt`, para forzar una reinstalación limpia de dependencias.

**Error `No such file or directory: .../data/base_datos_cartera_sintetica.csv`**

Este error ya no debería ocurrir: los datos de ejemplo se generan **en
memoria** dentro de `app.py` (misma lógica y semilla que
`data/generar_datos_sinteticos.py`), por lo que la app no depende de que ese
archivo exista físicamente en el repositorio. La carpeta `data/` se conserva
únicamente como referencia legible y para quien quiera regenerar el ejemplo
por su cuenta; si llegara a faltar en el repositorio, la app sigue
funcionando exactamente igual.

**Recomendación general:** para evitar que falten archivos al subir el
proyecto, usa `git` en vez de arrastrar archivos uno por uno en el navegador:

```bash
git init
git add .
git commit -m "Primera versión del optimizador de cartera"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<tu-repositorio>.git
git push -u origin main
```

Esto garantiza que **todos** los archivos y carpetas del proyecto (incluida
`data/` y `.streamlit/`) se suban de una sola vez, con la misma estructura
que tienes localmente.

---

## 8. Notas y limitaciones

- El modelo asume que los rendimientos mensuales por segmento son
  representativos de su comportamiento futuro esperado (supuesto estándar de
  Markowitz); no incorpora escenarios de estrés ni cambios estructurales.
- `limite_maximo_ejemplo_pct` en el archivo de ejemplo es un valor
  ilustrativo, no una política real de la arrendadora.
- La frontera eficiente y la simulación Monte Carlo son sensibles al periodo
  histórico seleccionado; se recomienda usar al menos 12–24 meses de
  historia por segmento.
