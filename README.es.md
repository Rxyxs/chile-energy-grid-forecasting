# ⚡ Pronóstico de la Red Eléctrica Chilena (SEN)

[ 🇺🇸 [English](README.md) ] | [ 🇨🇱 Español ]

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-forecaster-0193B0)
![XGBoost](https://img.shields.io/badge/XGBoost-comparaci%C3%B3n%20de%20estabilidad-EB0028)
![Optuna](https://img.shields.io/badge/Optuna-tuning%20de%20hiperpar%C3%A1metros-6A5ACD)
![Polars](https://img.shields.io/badge/Polars-feature%20engineering-CD792C)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-30%20passing-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Pronóstico horario multi-target del Sistema Eléctrico Nacional (SEN) de Chile — generación solar, generación eólica, demanda y costo marginal (precio spot) — para 5 nodos reales de 220kV, con modelos LightGBM ajustados vía Optuna, validados walk-forward contra baselines naive/estacional, features meteorológicas reales (radiación solar, viento, temperatura) que cierran la brecha causal entre el clima y la generación renovable, una comparación de estabilidad vía Rolling-Origin CV contra XGBoost, un motor de pronóstico recursivo multi-step, y un dashboard de monitoreo en Streamlit.

## 1. Contexto de negocio

El SEN chileno está dominado por energías renovables en sus nodos del norte (una de las mayores irradiancias solares del mundo, en el desierto de Atacama) y cada vez más por generación eólica en el sur. Esto genera una dinámica de mercado real y bien documentada: alta generación renovable en un nodo empuja su costo marginal hacia cero al desplazar generación térmica cara, mientras que un pico de demanda con baja generación renovable lo empuja fuertemente al alza. Pronosticar generación, demanda y precio con anticipación — a nivel de nodo, no solo a nivel de sistema — es lo que le permite a un generador, a un gran consumidor industrial (la minería es una carga relevante en el norte) o a un operador de mercado anticipar su exposición a costos y el riesgo de precios pico, en vez de reaccionar después de que ocurren. Pronosticar las cuatro series (no solo el precio) también importa operacionalmente: un operador de red que programa reservas necesita saber *cuánta* solar/eólica/demanda esperar, no solo cuánto va a costar.

## 2. Datos

El coordinador eléctrico de Chile (CEN) no expone una API pública simple y gratuita para datos históricos horarios por nodo, así que este proyecto genera **2 años (2024–2025) de datos horarios sintéticos** para 5 nodos/barras reales del SEN, construidos para reproducir la mecánica real del mercado y no solo ruido aleatorio:

| Nodo | Perfil | Cap. solar (MWh) | Cap. eólica (MWh) | Demanda base (MWh) |
|---|---|---|---|---|
| Crucero_220kV | Minería (carga casi plana, 24/7) | 220 | 40 | 180 |
| Cardones_220kV | Mixto | 200 | 60 | 90 |
| Quillota_220kV | Urbano (doble pico) | 110 | 70 | 140 |
| Alto_Jahuel_220kV | Urbano (doble pico) | 130 | 30 | 320 |
| Charrua_220kV | Industrial, alta eólica | 70 | 140 | 150 |

Decisiones clave de modelado en el generador (`src/data/fetch_energy_data.py`):
- **El clima va primero, la generación se deriva de él -- no al revés.** El generador calcula señales meteorológicas físicas por nodo (`ghi_w_m2`: irradiancia solar horizontal global, W/m²; `wind_speed_ms`; `temperature_c`), cada una reflejando la geografía real de Chile (los nodos del desierto de Atacama tienen algunos de los mayores GHI de cielo despejado del mundo; los nodos del sur son más fríos y nubosos), y **la generación solar/eólica se deriva matemáticamente de ellas** -- solar como fracción lineal del GHI de cielo despejado, eólica vía una curva de potencia de turbina simplificada (velocidad de cut-in/nominal, no un exponente arbitrario) -- en vez de calcularse como un proceso separado y desconectado.
- **Solar** sigue una ventana de luz diurna estacional (hemisferio sur: días más largos en diciembre, más cortos en junio) y una curva diurna en forma de campana, modulada por un factor de nubosidad diario -- todo capturado en `ghi_w_m2`.
- **Eólica**: la velocidad de viento es un camino aleatorio autocorrelado con reversión a la media y cambios de régimen ocasionales ("frentes de viento"), sin patrón diurno; la generación sigue una curva de potencia (cero bajo cut-in, creciente con la velocidad, plana en capacidad nominal por encima).
- **Demanda** usa una forma horaria específica por perfil (minería casi plana; urbano con un claro doble pico mañana/noche), un descuento de fin de semana/feriado (vía el calendario chileno de la librería `holidays`), una tendencia de crecimiento de 3%/año, y un efecto HVAC modesto por temperatura (consumo extra fuera de una zona de confort de 18-24°C) -- deliberadamente secundario frente a la forma horaria/calendario, que sigue siendo el driver dominante de la demanda, igual que en la realidad.
- **Costo marginal** se deriva de la *demanda residual* (demanda menos solar y eólica), siguiendo la lógica real de orden de mérito, más picos ocasionales de estrés de oferta/restricciones de transmisión. Los precios cercanos a cero durante alta generación solar y baja demanda residual son un fenómeno real y documentado en el norte de Chile, no un artefacto del generador.

Los datos crudos y las features procesadas están excluidos del control de versiones (`.gitignore`) y se regeneran a demanda (ver [Cómo ejecutar](#8-cómo-ejecutar)).

## 3. Arquitectura

```mermaid
flowchart LR
    A["fetch_energy_data.py\ngenerador sintético del SEN\n+ clima (GHI/viento/temp)"] -->|"data/raw/sen_hourly_data.csv"| B["build_features.py\nfeature engineering en Polars\n+ lags de clima"]
    B -->|"data/processed/sen_features.parquet"| C["train_forecaster.py\ntuning Optuna + LightGBM\nx4 targets + baselines"]
    B --> G["rolling_stability.py\nRolling-Origin CV\nLightGBM vs XGBoost"]
    C -->|"forecaster_&lt;target&gt;.joblib x4\nforecaster_metrics.json"| D["forecast.py\nCLI de pronóstico recursivo"]
    C --> E["dashboard.py\nStreamlit + Plotly"]
    D --> E
```

## 4. Metodología

**Feature engineering** (`src/features/build_features.py`, Polars, calculado **por nodo** vía `.over("node")` para que filas de distintas barras nunca se mezclen dentro de una misma ventana):
- Lags de 1h, 24h y 168h (1 semana) para solar, eólica, demanda y precio.
- **Lags meteorológicos** de 1h, 3h, 6h y 24h para `ghi_w_m2` (irradiancia solar), `wind_speed_ms` y `temperature_c` (`add_weather_lag_features`) — horizontes más cortos que los de los targets, porque el clima se autocorrelaciona y decae más rápido hora a hora que un patrón semanal de demanda, así que un lag de 168h aportaría poco. Igual que con los 4 targets, el valor crudo de la misma hora se excluye del set de features (`get_feature_columns`) — solo se usan lags, porque este proyecto simula observaciones históricas, no un pronóstico NWP que legítimamente conocería el clima de horas futuras por adelantado.
- Media y desvío móvil sobre ventanas de 6h y 24h — calculados sobre `shift(1)` para que la ventana nunca incluya la fila que se está prediciendo.
- Codificación cíclica seno/coseno de hora del día y mes, evitando el salto artificial 23→0 / dic→ene de un entero directo.
- Features de calendario: día de la semana, indicador de fin de semana, y feriados legales chilenos (incluyendo feriados móviles como Viernes Santo, vía la librería `holidays`).

**Modelo, tuning y validación** (`src/models/train_forecaster.py`): un `LGBMRegressor` por target — solar, eólica, demanda y precio comparten exactamente el mismo set de features (los valores crudos de la misma hora de los 4 targets, más las 3 variables meteorológicas crudas, siempre se excluyen; en un despliegue real tampoco se conocería la demanda o generación actual con certeza, solo sus lags y estadísticas móviles). Para cada target, **Optuna** (`tune_hyperparameters`) busca learning rate, profundidad/hojas del árbol, submuestreo y regularización L1/L2 sobre 20 trials, minimizando el WAPE walk-forward en 3 folds; los mejores hiperparámetros se re-validan luego con un pase completo de 5 folds walk-forward para el reporte final.

La validación usa `TimeSeriesSplit` aplicado sobre **timestamps únicos** (`src/models/validation.py`), no sobre las filas crudas del panel — con 5 nodos compartiendo cada hora, dividir solo por orden de fila podría dejar la hora *t* de un nodo en entrenamiento mientras la hora *t-1* de otro nodo cae en test, lo cual sigue siendo lookahead bias aunque el índice de fila sea "posterior". Dividir sobre el eje temporal mueve los datos de todos los nodos de un mismo bloque de tiempo juntos. Los baselines (`src/models/baselines.py`) se evalúan sobre exactamente los mismos folds y la misma métrica WAPE, así que la comparación LightGBM-vs-baseline de abajo es directa y honesta.

**Rolling-Origin CV** (`src/models/rolling_stability.py`, `validation.get_rolling_origin_folds`): una segunda estrategia de validación, distinta del walk-forward expansivo de arriba. Cada fold entrena sobre una ventana de tamaño **fijo** (180 días por defecto) y evalúa sobre los 14 días siguientes, deslizándose hacia adelante — a diferencia del walk-forward expansivo (donde el train set de cada fold sigue creciendo), acá todos los folds ven la misma cantidad de historia reciente. Eso aísla específicamente la **estabilidad temporal**: si el error de un modelo varía mucho fold a fold con un tamaño de ventana de entrenamiento constante, eso es evidencia de inestabilidad real, no solo de que "el fold tardío tenía más datos". Se usa para comparar LightGBM contra **XGBoost** (listado en `requirements.txt` desde el inicio del proyecto pero nunca usado en ningún lado hasta ahora) vía el coeficiente de variación (CV = desvío/media) del WAPE entre folds de cada modelo — ver §5.3.

El error se reporta como **WAPE** (Weighted Absolute Percentage Error: `sum(|error|) / sum(|real|)`) en vez de MAPE, porque `solar_generation_mwh` es exactamente 0 todas las noches, lo que haría indefinido un error porcentual punto a punto en cada hora nocturna.

**Pronóstico multi-step hacia adelante** (`src/models/forecast.py`): predice recursivamente N horas hacia adelante por nodo. Como los 4 targets comparten un mismo set de features construido solo con lags/estadísticas móviles (nunca valores de la misma hora), un solo vector de features por hora futura alcanza para predecir los 4 a la vez — no hace falta decidir "cuál target pronosticar primero". Cada hora predicha se retroalimenta al buffer de historia antes de avanzar a la siguiente, exactamente como funcionaría en producción, donde el futuro real todavía no existe.

## 5. Resultados

### 5.1 Validación walk-forward: LightGBM (ajustado con Optuna, con clima) vs. baselines

Validación walk-forward, 5 folds cada uno, 48 features compartidas (36 previas + 12 lags de clima), LightGBM (ajustado con Optuna) vs. dos baselines evaluados sobre los mismos folds — **naive** (persiste el valor de hace 1h) y **estacional** (persiste el valor de la misma hora, 24h atrás). Negrita marca el mejor real de los tres por fila — no siempre LightGBM, a propósito (ver el hallazgo honesto abajo):

| Target | WAPE LightGBM | WAPE naive | WAPE estacional | MAE LightGBM |
|---|---|---|---|---|
| Generación solar | **3,60%** | 26,95% | 19,46% | 1,21 MWh |
| Generación eólica | 10,31% | **9,93%** | 44,89% | 2,13 MWh |
| Demanda | **2,54%** | 4,79% | 5,60% | 4,28 MWh |
| Costo marginal | **5,95%** | 10,26% | 12,08% | $5,27/MWh |

Todos los números provienen directamente de ejecutar `python -m src.models.train_forecaster` de punta a punta (semilla 42, 87.720 filas sobre 5 nodos × 17.544 horas, 20 trials de Optuna por target). LightGBM supera claramente a ambos baselines en solar, demanda y precio; el WAPE de solar bajó de 4,57% a 3,60% respecto de la línea base previa al clima, la ganancia más grande de esta ronda de trabajo.

**Hallazgo honesto, reportado tal cual en vez de ajustado hasta que se viera bien: en eólica, la persistencia naive de 1 hora (9,93% WAPE) ahora supera levemente a LightGBM (10,31%), incluso con features meteorológicas reales y tuning de Optuna.** Esto invierte la hipótesis que el propio README anterior planteaba -- "un paso siguiente real sería agregar features de pronóstico meteorológico [para cerrar la brecha estrecha de LightGBM vs. naive en eólica]" -- y la ablación controlada de §5.2 muestra *por qué* esa hipótesis no se sostiene: `wind_generation_mwh` acá es cercano a un camino aleatorio puro con reversión a la media (reversión muy débil, 2%/hora; autocorrelación hora a hora de ~0,99 en el proceso de velocidad de viento subyacente), un régimen donde la persistencia a 1 paso está cerca del mejor pronóstico puntual posible desde el punto de vista de la teoría de la información, sin importar qué otras features estén disponibles — una propiedad real y bien documentada del pronóstico eólico a horizontes cortos, no un bug de este pipeline. Solar no tiene este problema porque tiene una estructura determinística fuerte (un ciclo diurno/estacional) que la persistencia no puede explotar pero que la irradiancia y las features de calendario sí.

### 5.2 Aislando el efecto real del clima: una ablación controlada (`notebooks/02_Weather_Augmented_Rolling_CV.ipynb`)

La mejora de §5.1 (solar 4,57%→3,60%) confunde dos cambios a la vez: las nuevas features meteorológicas, *y* el re-tuning de Optuna sobre el espacio de features más grande. El notebook aísla solo las features meteorológicas, manteniendo los hiperparámetros **fijos** en una comparación con/sin:

| Target | WAPE sin clima | WAPE con clima | Delta |
|---|---|---|---|
| Generación solar | 3,79% | 3,69% | **+0,10 pts** |
| Generación eólica | 10,55% | 10,55% | +0,003 pts (nulo) |
| Demanda | 2,55% | 2,55% | −0,01 pts |
| Costo marginal | 6,18% | 6,26% | −0,08 pts |

**El efecto aislado es real para solar, prácticamente nulo para eólica, y levemente negativo para demanda/precio** — mucho menor de lo que el número titular de §5.1 sugería, y esa diferencia entre las dos tablas *es en sí misma el hallazgo*: una comparación antes/después ingenua que cambia features y re-tunea hiperparámetros al mismo tiempo sobreestima cuánto crédito corresponde específicamente a las features. Eólica no muestra beneficio medible porque los propios lags de `wind_generation_mwh` ya codifican casi toda la información que los lags de `wind_speed_ms` podrían aportar (la generación es una función casi determinística de la velocidad de viento acá) — los dos sets de features son redundantes, no complementarios. Demanda y precio empeoraron levemente con más features y sin re-tuning — un efecto estándar y bien conocido de agregar dimensionalidad sin ajustar la regularización para ella.

### 5.3 Rolling-Origin CV: estabilidad temporal de LightGBM vs. XGBoost

8 folds, ventana fija de entrenamiento de 180 días / evaluación de 14 días, deslizándose hacia adelante (`src/models/rolling_stability.py` — XGBoost, listado en `requirements.txt` desde el inicio de este proyecto pero nunca usado en ningún lado hasta ahora):

| Target | WAPE medio LightGBM (CV) | WAPE medio XGBoost (CV) |
|---|---|---|
| Solar | 4,10% (0,100) | 4,21% (0,115) |
| Eólica | 10,82% (0,142) | 10,99% (0,137) |
| Demanda | 2,51% (0,046) | 2,51% (0,040) |
| Costo marginal | **6,87% (0,051)** | 7,89% (0,133) |

CV = coeficiente de variación (desvío/media) del WAPE entre folds — más bajo significa más estable a través del tiempo. Tres targets quedan parejos entre ambos modelos, pero **en el costo marginal, el WAPE de XGBoost oscila entre 6,72% y 9,96% según qué bloque de 14 días le toque evaluar (CV 0,133), mientras que LightGBM se mantiene en una banda mucho más angosta de 6,18%-7,32% (CV 0,051)** — aproximadamente 2,6x más estable. Este es exactamente el tipo de problema que la validación walk-forward expansiva (§5.1, la que se usa para el modelo desplegado) no puede exponer estructuralmente: con un train set que sigue creciendo, un fold malo aislado se diluye en el promedio en vez de destacar. Un modelo que en promedio está bien pero es poco confiable fold a fold es un riesgo real de producción que solo una comparación de ventana fija como esta expone — una razón concreta, respaldada por datos, para preferir LightGBM sobre XGBoost específicamente para el target de precio, no solo una preferencia general.

El panel histórico del dashboard de Streamlit muestra precio real vs. pronosticado sobre el set de test fuera de muestra del último fold — predicciones genuinas, no ajuste dentro de muestra — y un panel separado corre el pronosticador recursivo multi-step real sobre horas que ni siquiera existen en el dataset.

## 6. Ejemplo de pronóstico multi-step

```bash
python -m src.models.forecast --horizon 12 --nodes Crucero_220kV
```

Salida real (nodo de perfil minero, pronóstico desde medianoche local): la generación solar sube de 0 a 154,7 MWh entre las 00:00 y las 11:00 a medida que amanece, y el costo marginal cae de $112,39/MWh a $71,04/MWh en la misma ventana a medida que la solar creciente desplaza generación cara — la dinámica de orden de mérito del generador de datos, recuperada puramente de las predicciones recursivas del modelo, no hardcodeada.

**Una limitación real que la propia salida del pronosticador recursivo expone**: la generación eólica se mantiene prácticamente plana (~6,9 MWh) durante las 12 horas del ejemplo de arriba. No es una falla del modelo -- es `forecast.py` persistiendo hacia adelante el último valor *conocido* de clima (ver `forecast_node`, `last_known_weather`), porque este proyecto simula observaciones históricas de clima, no un pronóstico NWP para horas futuras. La generación solar sigue evolucionando correctamente porque su driver dominante (la codificación cíclica de hora del día/mes) se recalcula exactamente para cada timestamp futuro sin importar la persistencia del clima, pero la eólica -- que depende casi por completo del valor de clima persistido -- visiblemente pierde su propia dinámica de camino aleatorio más allá del primer paso. Documentado acá en vez de ocultarlo: un despliegue en producción necesitaría pronósticos NWP reales de viento para el horizonte recursivo, no persistencia.

## 7. Stack tecnológico

Python · Polars · pandas · NumPy · LightGBM · XGBoost · Optuna · scikit-learn (`TimeSeriesSplit`) · Streamlit · Plotly · `holidays` · Jupyter/matplotlib (notebook) · pytest (30 tests: chequeos de fuga de features, correctitud de lags meteorológicos, invariantes de partición de folds tanto para walk-forward como Rolling-Origin CV, corrección de baselines, validación end-to-end del pronóstico recursivo, smoke test del dashboard vía `streamlit.testing.v1.AppTest`)

## 8. Cómo ejecutar

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m src.data.fetch_energy_data      # generar datos sintéticos del SEN (targets + clima)
python -m src.features.build_features     # construir features de lag/rolling/cíclicas/clima
python -m src.models.train_forecaster      # tuning Optuna + entrenar/validar walk-forward, los 4 targets
# opcional: --trials N (default 20) --splits N (default 5)

python -m src.models.rolling_stability     # Rolling-Origin CV, LightGBM vs XGBoost, los 4 targets

python -m src.models.forecast --horizon 24 --nodes all   # pronóstico recursivo a N horas (CLI)
streamlit run src/app/dashboard.py                        # levantar el dashboard de monitoreo

python -m pytest tests/ -v                                 # correr la suite de tests

jupyter nbconvert --to notebook --execute --inplace notebooks/02_Weather_Augmented_Rolling_CV.ipynb
```

## 9. Estructura del proyecto

```
src/
  data/       fetch_energy_data.py    generador de datos horarios sintéticos del SEN + clima (GHI/viento/temp)
  features/   build_features.py       features de lag/rolling/cíclicas/calendario/lags de clima en Polars
  models/     validation.py           folds walk-forward + folds Rolling-Origin + métrica WAPE
              baselines.py            métricas de referencia naive / estacional
              train_forecaster.py     tuning Optuna + LightGBM, 4 targets
              rolling_stability.py    Rolling-Origin CV, estabilidad temporal LightGBM vs XGBoost
              forecast.py             CLI de pronóstico recursivo multi-step
  app/        dashboard.py            dashboard de monitoreo + pronóstico en vivo, en Streamlit
notebooks/    02_Weather_Augmented_Rolling_CV.ipynb   ablación de features de clima + análisis de estabilidad
tests/                                pytest: features, validación, baselines, forecast, dashboard,
                                       estabilidad rolling, exclusión de columnas de features
```

## 10. Autor

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE)
