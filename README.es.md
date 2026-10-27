# ⚡ Pronóstico de la Red Eléctrica Chilena (SEN)

[[English](README.md) ] | [Español ]

![CI](https://github.com/Rxyxs/chile-energy-grid-forecasting/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-forecaster-0193B0)
![XGBoost](https://img.shields.io/badge/XGBoost-comparaci%C3%B3n%20de%20estabilidad-EB0028)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP%20%2B%20loss%20Huber-EE4C2C?logo=pytorch&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-almac%C3%A9n%20de%20m%C3%A9tricas-FFF000)
![Optuna](https://img.shields.io/badge/Optuna-tuning%20de%20hiperpar%C3%A1metros-6A5ACD)
![Polars](https://img.shields.io/badge/Polars-feature%20engineering-CD792C)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-extracci%C3%B3n%20de%20gr%C3%A1ficos%20vectoriales-1F6FEB)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Conformal Prediction](https://img.shields.io/badge/Predicci%C3%B3n%20Conformal-intervalos%20CQR-4a3aa7)
![pytest](https://img.shields.io/badge/pytest-87%20passing-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Pronóstico horario del Sistema Eléctrico Nacional (SEN) de Chile sobre **cuatro años de datos horarios reales (2022-2025)**: el costo marginal de 6 barras reales de 220 kV, de Antofagasta a Puerto Montt, y la generación solar, la generación eólica y la demanda del sistema completo. Las series horarias no vienen de una API (la del Coordinador exige una cuenta institucional): se leen de los gráficos vectoriales de sus informes diarios públicos en PDF, se calibran día a día contra los números que imprime cada informe y se validan contra dos series mensuales oficiales independientes. Encima de eso: LightGBM ajustado con Optuna y validado walk-forward contra baselines ingenuos, variables de clima reales (ERA5), Rolling-Origin CV contra XGBoost, una MLP en PyTorch, intervalos de predicción conformales, un pronóstico recursivo multi-step y un dashboard en Streamlit.

## 1. Contexto de negocio

El SEN es una línea de 3.000 km. El norte (Atacama) tiene de las radiaciones solares más altas del mundo; el sur tiene hidro y viento; la transmisión entre ambos es limitada. Eso crea una dinámica de mercado real y documentada: a mediodía el excedente solar del norte lleva su costo marginal a cero mientras las barras más al sur siguen pagando térmica e hidro, y en la tarde el precio salta en todas partes cuando se pone el sol. Pronosticar el precio **por barra**, no un único precio del sistema, es lo que permite a un generador, a un gran consumidor industrial (la minería es una carga mayor en el norte) o a un trader anticipar su exposición de costo. Pronosticar la solar, la eólica y la demanda del sistema importa por sí mismo para programar reservas, y además alimenta el modelo de precio.

## 1.1 Resultados clave

| Métrica | Resultado | Qué significa |
|---|---|---|
| Series horarias reconstruidas de los gráficos del PDF | **1.397 días** con las 6 barras; residuo mediano de calibración **0,04 USD/MWh** | El gráfico de cada día se calibra contra los promedios diarios impresos en el mismo informe |
| WAPE de la demanda (sistema) | **1,20%** vs. 2,50% del ingenuo | La mitad del error de la persistencia de 1 hora |
| WAPE de la solar (sistema) | **6,26%** vs. 22,05% del ingenuo, 11,34% la misma hora de ayer | La hora del día y la producción reciente explican la mayor parte |
| WAPE de la eólica (sistema) | **9,01%** vs. 11,80% del ingenuo | La persistencia es un baseline difícil para el viento |
| WAPE del costo marginal, 6 barras | **17,88%** mezclado con la persistencia (18,71% LightGBM solo) vs. 21,88% del ingenuo | Los precios reales son mucho más difíciles que los del panel simulado anterior (8,77%) |
| Costo marginal en Puerto Montt | LightGBM 22,75%, mezcla **20,34%**, ingenuo 20,82% | Solo, el modelo pierde contra la persistencia en la barra desacoplada del sistema; un peso de mezcla por barra elegido en validación lo arregla (§5.1) |
| MLP de PyTorch sobre el precio | 22,41% (Swish, elegida en validación) vs. 18,71% de LightGBM ajustado | Con el panel simulado la MLP ganaba; con precios reales no le gana a la persistencia |
| Intervalos de predicción al 90% | Todos los folds de todos los targets cubiertos **89-91%** con inferencia conformal adaptativa | Un margen conformal fijo cubre la solar 73% en su peor fold: se calibra en una estación y se usa en la siguiente (§5.5) |

## 2. Datos: cuatro años de datos horarios reales, leídos de gráficos en PDF

El Coordinador Eléctrico Nacional (CEN) publica el costo marginal horario por barra y la generación horaria por tecnología, pero su API exige una cuenta institucional y su portal de datos bloquea las descargas automatizadas. También publica cada día, sin cuenta, un PDF llamado *Resumen Ejecutivo de Operación*. Sus gráficos son **dibujos vectoriales, no imágenes**: cada línea y cada área apilada guarda las coordenadas exactas con que se dibujó. `src/data/cen_daily_reports.py` descarga los informes de 2022-2025 (están en línea 1.413 de 1.461 días) y lee de ellos dos series horarias:

| Serie | Fuente | Granularidad | Uso |
|---|---|---|---|
| Costo marginal en Crucero, Cardones, Pan de Azúcar, Quillota, Charrúa y Puerto Montt 220 kV | Gráfico "Costo Marginal Real Preliminar" del informe: una línea de 24 puntos por barra. Entre julio y agosto de 2025 los informes pasan al costo marginal *en línea*, una línea de 96 puntos por barra (cada 15 minutos), que se promedia por hora | Horaria, real | Target de precio, por barra |
| Generación por tecnología: térmica, hidráulica, solar, eólica, baterías | Gráfico "Generación horaria bruta por tecnología" del informe: áreas apiladas | Horaria, real | Targets de solar y eólica; su suma (generación bruta) es el target de demanda |
| Radiación solar (GHI), viento a 100 m, temperatura | Open-Meteo (reanálisis ERA5) en las coordenadas de cada barra | Horaria, real | Variables de clima |

**Calibración.** Ninguno de los dos gráficos tiene una escala explícita confiable (la grilla está ajustada a píxeles), así que cada día se calibra contra números que el mismo informe imprime como texto:

- La leyenda del costo marginal imprime el **promedio diario** de cada barra. Con 6 a 8 barras y una recta de 2 parámetros (píxeles → USD/MWh), el ajuste queda sobredeterminado y su residuo mide la reconstrucción. Residuo mediano de la peor barra: **0,04 USD/MWh**; percentil 99: 0,23 (los promedios impresos están redondeados a 0,1).
- La tabla de generación imprime la **energía diaria** (GWh) de cada tecnología. Una sola escala (MW por punto) tiene que reproducir las cuatro tecnologías grandes a la vez: error mediano de la peor tecnología **0,07%**, percentil 95 2,5%. La forma horaria sale del gráfico y cada tecnología se ajusta después a su energía diaria (las baterías, 1-2% de la energía, se dibujan de forma inconsistente: se ajustan pero no se usan para calibrar).

Llegar a eso exigió manejar cinco formatos de informe y los días de cambio de horario (gráficos de 23 o 25 horas): la leyenda de barras cambió nombres y diagramación, los informes de comienzos de 2022 dibujan la eólica solo como una línea sobre el apilado, algunas áreas vienen como varios subtrazos en un mismo dibujo (las baterías descargan de madrugada y de noche, con un hueco al mediodía), el PDF omite los vértices colineales de los tramos planos, y los informes de fines de 2025 suavizan las áreas con curvas Bézier. `tests/test_cen_daily_reports.py` fija esos casos con entradas sintéticas.

**Validación contra fuentes independientes.** Dos series mensuales públicas no se usaron para construir nada, así que sirven para revisarlo:

![Reconstrucción vs. series mensuales oficiales](data/processed/real_reconstruction_check.png)

- **Energía mensual vs. la generación oficial del SEN (datos.gob.cl):** el total queda dentro de 3% (en general 1-2%) y la eólica dentro de 5%. La solar de los gráficos queda sistemáticamente 2-8% *bajo* la serie oficial: las dos fuentes no cuentan exactamente las mismas centrales.
- **Costo marginal mensual de Crucero vs. los reportes mensuales de la CNE:** de junio de 2022 a junio de 2024 el promedio de las horas reconstruidas calza con la cifra de la CNE con una diferencia de alrededor de 0,5 USD/MWh. **Desde julio de 2024 las dos publicaciones divergen**: el informe diario da 9-30 USD/MWh menos que la CNE todos los meses. La reconstrucción sigue reproduciendo el valor diario que el propio informe imprime en otra página (45,34 vs. 45,33 el 10 de julio de 2024; 64,00 vs. 63,98 el 15 de octubre de 2024), así que la diferencia está entre las dos publicaciones, no en la extracción. Estos archivos no permiten saber cuál cambió de definición; el target de precio de este proyecto es el del informe diario.

**Cómo es el panel.** Ninguna fuente pública trae generación ni demanda *por barra*, así que el panel no las inventa: junto a las 6 barras tiene un séptimo nodo, `SEN`, que lleva los targets de sistema (solar, eólica, demanda) con el clima promedio de las 6 barras. Las filas de cada barra también llevan las series del sistema, para que sus rezagos entren al modelo de precio (el orden de mérito depende de la demanda residual del sistema, no de la de la barra). `real_panel.rows_for_target` decide qué filas entrenan y evalúan cada target: el nodo `SEN` para solar, eólica y demanda, las 6 barras para el precio, así ninguna métrica cuenta siete veces la misma hora del sistema.

**Limpieza.** Las horas en que la generación total cae bajo la mitad de la mediana de su día quedan como faltantes (la hora que no existe el día que empieza el horario de verano, que el gráfico dibuja en 0, y el apagón nacional del 25 de febrero de 2025). Los días sin informe o sin gráfico utilizable quedan faltantes, sin interpolar: el 96,6% de las horas tiene datos del sistema y el 96,1% tiene precios. Los PDF (alrededor de 1 GB) no se versionan; las series horarias extraídas en `data/real/` sí.

`generate_energy_data(use_real=False)` sigue produciendo el panel completamente simulado original (5 barras, 2024-2025), que se conserva para los tests que necesitan un proceso generador conocido.

![Costo marginal horario real por barra](data/processed/real_price_profile_by_bar.png)

**El precio se separa por geografía a mediodía.** Entre las 10:00 y las 16:00, Crucero, Cardones y Pan de Azúcar quedan bajo 1 USD/MWh en el 70-75% de las horas (excedente solar del norte), Quillota y Charrúa en el 53-56%, y Puerto Montt solo en el 29%. La demanda residual del sistema correlaciona 0,64-0,73 con el precio en cinco barras y 0,20 en Puerto Montt (`notebooks/01_eda_seasonality_analysis.ipynb`): una sola variable del sistema no fija el precio de una barra que la transmisión aísla.

## 3. Arquitectura

```mermaid
flowchart LR
    R["cen_daily_reports.py\nPDF diarios del CEN -> precio horario\n(6 barras) + generación por tecnología"] -->|"data/real/*.csv.gz"| A["real_panel.py\n6 barras + nodo SEN\n+ clima Open-Meteo"]
    A -->|"data/raw/sen_hourly_data.csv"| B["build_features.py\nfeature engineering con Polars\n+ rezagos de clima"]
    B -->|"data/processed/sen_features.parquet"| C["train_forecaster.py\ntuning Optuna + LightGBM\nx4 targets + baselines"]
    B --> G["rolling_stability.py\nRolling-Origin CV\nLightGBM vs XGBoost"]
    B --> H["torch_forecaster.py\nMLP PyTorch + loss Huber\nReLU/GELU/Swish"]
    C -->|"forecaster_&lt;target&gt;.joblib x4\nforecaster_metrics.json"| D["forecast.py\npronóstico recursivo multi-step\nprimero SEN, después barras"]
    C --> E["dashboard.py\nStreamlit + Plotly"]
    C -->|"hiperparámetros ajustados"| J["conformal.py\nintervalos CQR\nx4 targets"]
    H --> I["generate_torch_report.py\ngráficos + metrics.duckdb"]
    G --> I
    J -->|"conformal_&lt;target&gt;_{lower,upper}.joblib\nconformal_metrics.json"| D
    J --> E
    D --> E
```

## 4. Metodología

**Feature engineering** (`src/features/build_features.py`, Polars, calculado **por nodo** con `.over("node")`, así las filas de nodos distintos nunca se mezclan en una misma ventana):
- Rezagos de 1h, 24h y 168h (1 semana) para solar, eólica, demanda y precio. En las filas de cada barra, los rezagos de solar, eólica y demanda son los del sistema: el modelo de precio ve la demanda residual reciente del sistema.
- **Rezagos de clima** de 1h, 3h, 6h y 24h para `ghi_w_m2`, `wind_speed_ms` y `temperature_c`. El clima crudo de la misma hora se excluye (`get_feature_columns`): el proyecto usa observaciones históricas, no un pronóstico meteorológico que legítimamente conocería la hora futura.
- Media y desvío móviles de 6h y 24h, calculados sobre `shift(1)` para que la ventana nunca incluya la fila que se pronostica.
- Codificación seno/coseno de la hora del día y del mes; día de la semana, fin de semana y feriados legales de Chile (incluidos los movibles, con `holidays`).

**Modelo, tuning y validación** (`src/models/train_forecaster.py`): un `LGBMRegressor` por target. Para cada uno, **Optuna** busca tasa de aprendizaje, profundidad/hojas, submuestreo y regularización L1/L2 en 20 trials, minimizando el WAPE walk-forward en 3 folds; los mejores hiperparámetros se revalidan en 5 folds walk-forward para el reporte. Cada target se entrena y evalúa solo en sus filas (`rows_for_target`).

La validación usa `TimeSeriesSplit` sobre **timestamps únicos** (`src/models/validation.py`), no sobre filas del panel: con 7 nodos por hora, partir por orden de filas podría dejar la hora *t* de un nodo en entrenamiento y la hora *t-1* de otro en test. Los baselines (`src/models/baselines.py`) se evalúan en los mismos folds con la misma métrica.

**Rolling-Origin CV** (`src/models/rolling_stability.py`): cada fold entrena con una ventana **fija** de 180 días y evalúa los 14 días siguientes, deslizándose hacia adelante, así cada fold ve la misma cantidad de historia; el coeficiente de variación (desvío/media) del WAPE entre folds mide la estabilidad temporal, LightGBM vs. **XGBoost**.

El error se reporta como **WAPE** (`sum(|error|) / sum(|real|)`) y no como MAPE, porque la solar es exactamente 0 cada noche y el precio es exactamente 0 a mediodía en el norte buena parte del año, lo que dejaría indefinido un porcentaje por punto.

**Pronóstico multi-step** (`src/models/forecast.py`): recursivo, cada hora pronosticada vuelve al buffer histórico antes de la siguiente. En el panel real primero se pronostica el sistema (solar, eólica y demanda en el nodo `SEN`) y después el precio de cada barra con esa trayectoria del sistema, en vez de que cada barra adivine la solar del país desde su propio clima.

## 5. Resultados

### 5.1 Validación walk-forward: LightGBM (ajustado con Optuna) vs. baselines

5 folds sobre 2022-2025, LightGBM (ajustado con Optuna) vs. dos baselines evaluados en los mismos folds: **ingenuo** (el valor de hace 1h) y **estacional ingenuo** (la misma hora de hace 24h):

| Target | WAPE LightGBM | WAPE ingenuo | WAPE estacional | MAE LightGBM |
|---|---|---|---|---|
| Solar (sistema) | **6,26%** | 22,05% | 11,34% | 127 MW |
| Eólica (sistema) | **9,01%** | 11,80% | 34,48% | 111 MW |
| Demanda (sistema) | **1,20%** | 2,50% | 4,47% | 116 MW |
| Costo marginal (6 barras) | **18,71%** | 21,88% | 39,06% | $12,44/MWh |

Todas las cifras salen de correr `python -m src.models.train_forecaster` de punta a punta (semilla 42, 20 trials de Optuna por target). LightGBM le gana a ambos baselines en todos los targets. El resultado de precio se reparte muy distinto por barra:

![WAPE del precio por barra: modelo vs. persistencia](data/processed/price_wape_by_bus.png)

| Barra (norte → sur) | WAPE LightGBM | WAPE ingenuo | WAPE estacional | LightGBM vs. ingenuo |
|---|---|---|---|---|
| Crucero | **17,20%** | 23,38% | 31,03% | −26% |
| Cardones | **17,41%** | 23,41% | 31,44% | −26% |
| Pan de Azúcar | **18,01%** | 23,10% | 34,36% | −22% |
| Quillota | **17,52%** | 20,73% | 36,15% | −15% |
| Charrúa | **18,59%** | 21,02% | 38,54% | −12% |
| Puerto Montt | 22,75% | **20,82%** | 57,47% | **+9%** |

**La ventaja del modelo se achica de norte a sur y se vuelve negativa en Puerto Montt.** En el norte, el colapso del precio a mediodía se puede anticipar con la solar del sistema y la hora del día, algo que la persistencia no ve venir. El precio de Puerto Montt casi no se relaciona con la demanda residual del sistema (correlación 0,20, §2), así que las variables que sostienen a las otras cinco barras no le sirven, y un modelo compartido entre barras le aplica el patrón del norte donde no corresponde. La subsección siguiente prueba qué lo arregla.

**Qué cambió respecto del panel simulado.** La versión anterior de este proyecto pronosticaba el precio con 8,77% de WAPE sobre un panel cuyos precios horarios eran simulados alrededor de un nivel mensual real. Con precios horarios reales el mismo pipeline llega a 18,71%, y su ventaja sobre la persistencia baja de 21% a 14%. Los precios reales tienen muchos ceros (el norte a mediodía), cambian de régimen (un 2022 con promedio de unos 100 USD/MWh, un 2024 bajo 50 en cinco de las seis barras) y se separan por geografía; el simulador no tenía nada de eso.

### Qué arregla Puerto Montt: reaccionar menos, no más información

`src/models/price_drivers.py` prueba, con los mismos folds walk-forward e hiperparámetros por defecto fijos, qué le podría faltar al modelo de precio:

| Variante | WAPE del precio, 6 barras | Puerto Montt |
|---|---|---|
| Un modelo para todas las barras (variables actuales) | 19,59% | 23,49% |
| + generación hidro y térmica del sistema (rezagos 1h, 24h, 168h) | 19,98% | 23,79% |
| + precios de las seis barras (rezagos 1h, 24h) | 19,54% | 23,19% |
| + ambas | 20,17% | 23,85% |
| Un modelo por barra, con ambas | 21,21% | 26,72% |
| Pronosticar el cambio respecto de hace 1 hora en vez del nivel | 19,35% | 23,11% |
| **Mezcla con la persistencia, un peso por barra** | **18,23%** | **20,29%** |
| Ingenuo (valor de hace 1 hora) | 21,77% | 20,67% |

**Más información no ayudó.** Las series horarias de hidro y térmica (ya leídas de los informes diarios y sin usar hasta ahora) y el estado de la red en los precios de las otras barras empeoran levemente todas las barras, y seis modelos separados, cada uno con un sexto de los datos, empeoran más. **Reaccionar menos sí.** La mezcla pronostica `w × modelo + (1 − w) × precio de hace 1 hora`, con un peso por barra elegido en el último 15% del entrenamiento de cada fold (con un modelo ajustado en el 85% anterior), nunca en el test. Los pesos elegidos siguen la geografía: 0,75-1,0 en el norte, 0,55-0,80 en Charrúa y 0-0,4 en Puerto Montt, donde el mejor pronóstico de la hora siguiente es sobre todo la hora actual.

Con los hiperparámetros ajustados por Optuna la mezcla ya es parte de `train_forecaster.py` (`src/models/blend.py`), y `forecast.py` y el dashboard aplican los pesos desplegados:

| Barra | LightGBM | Mezcla | Ingenuo | Peso del modelo desplegado |
|---|---|---|---|---|
| Crucero | 17,20% | **16,99%** | 23,38% | 0,90 |
| Cardones | 17,41% | **17,17%** | 23,41% | 0,90 |
| Pan de Azúcar | 18,01% | **17,77%** | 23,10% | 0,85 |
| Quillota | 17,52% | **16,87%** | 20,73% | 0,80 |
| Charrúa | 18,59% | **17,85%** | 21,02% | 0,80 |
| Puerto Montt | 22,75% | **20,34%** | 20,82% | 0,25 |
| **Las seis** | 18,71% | **17,88%** | 21,88% | |

La mezcla mejora todas las barras y convierte a Puerto Montt de una pérdida contra la persistencia (+9%) en una pequeña ganancia (−2%).

### 5.2 Aislando el efecto del clima: una ablación controlada (`notebooks/02_Weather_Augmented_Rolling_CV.ipynb`)

El notebook compara cada target **con y sin** los rezagos de clima, con hiperparámetros **fijos** (para no confundir la comparación con un reajuste):

| Target | WAPE sin clima | WAPE con clima | Delta |
|---|---|---|---|
| Solar (sistema) | 6,29% | 6,26% | +0,04 pts |
| Eólica (sistema) | 9,25% | 9,18% | +0,07 pts |
| Demanda (sistema) | 1,26% | 1,24% | +0,02 pts |
| Costo marginal (6 barras) | 19,87% | 19,75% | +0,12 pts |

**Con datos reales, el clima de seis subestaciones casi no aporta.** Todos los deltas están dentro de 0,12 puntos, y el del precio cambió de signo entre dos corridas de este proyecto (−0,19 puntos antes de agregar 13 días más de informes, +0,12 después): a este tamaño es ruido. En el panel simulado el clima ayudaba a la solar en 0,29 puntos y a la eólica en 0,54, pero ahí la generación se calculaba *desde* ese mismo clima, así que el clima traía información por construcción. Aquí los targets son la producción de todo el sistema, y seis puntos de clima son un proxy débil: la mejor barra correlaciona 0,84 con la solar del sistema pero solo 0,43 con su eólica (§2, notebook 01). Lo que movería estos números es un pronóstico meteorológico real para las horas siguientes, en los parques y no en las subestaciones.

### 5.3 Rolling-Origin CV: estabilidad temporal de LightGBM vs. XGBoost

8 folds, ventana fija de 180 días de entrenamiento / 14 de evaluación, deslizándose sobre el final de 2025 (`src/models/rolling_stability.py`):

| Target | LightGBM WAPE medio (CV) | XGBoost WAPE medio (CV) |
|---|---|---|
| Solar | **5,01%** (0,104) | 5,28% (**0,085**) |
| Eólica | **9,44% (0,091)** | 9,67% (0,100) |
| Demanda | **1,30% (0,107)** | 1,37% (0,119) |
| Costo marginal | **24,97% (0,129)** | 25,23% (0,137) |

CV = coeficiente de variación (desvío/media) del WAPE entre folds; más bajo significa más estable entre bloques de tiempo. **LightGBM es algo más preciso en los cuatro targets; cuál modelo es más estable no es un resultado robusto.** En precio, esta corrida le da a LightGBM el CV más bajo (0,129 vs. 0,137), mientras que la corrida de la misma comparación en el notebook se lo da a XGBoost (0,124), porque la construcción multihilo de los histogramas de XGBoost no es reproducible bit a bit, y una corrida anterior con 13 días menos también dejaba adelante a XGBoost. En el panel simulado LightGBM era 2,6 veces más estable; con datos reales los dos son indistinguibles. Con una ventana fija de 180 días el error de precio (20-32% por fold) es mayor que en el walk-forward expansivo de §5.1: medio año de historia no contiene los regímenes de precio que sí contiene la historia completa.

El panel histórico del dashboard muestra el precio real vs. el pronosticado sobre el test fuera de muestra del último fold (predicciones realmente retenidas, no ajuste dentro de muestra), y un panel aparte corre el pronóstico recursivo sobre horas que no están en los datos.

### 5.4 Un tercer enfoque de modelado: MLP en PyTorch vs. LightGBM/XGBoost, y ReLU vs. GELU vs. Swish

`src/models/torch_forecaster.py` agrega una MLP densa (dos capas ocultas, 64→32 unidades) sobre las mismas variables de rezagos y ventanas móviles que usan los árboles, con los mismos folds walk-forward y la misma métrica, una **loss Huber propia** (cuadrática para residuos chicos, lineal sobre `delta=1.0`, robusta a picos de precio) y una comparación controlada **ReLU / GELU / Swish** (arquitectura, optimizador y semilla fijos).

Walk-forward, 5 folds, target `marginal_cost_usd_mwh` (`python -m src.models.generate_torch_report`, semilla 42, CPU; las filas de árboles usan parámetros por defecto, no los ajustados con Optuna de §5.1):

| Modelo | WAPE | MAE | Latencia de inferencia (ms / 1k filas) |
|---|---|---|---|
| Ingenuo (persistencia 1h) | 21,88% | $14,39/MWh | -- |
| Estacional ingenuo (24h) | 39,06% | $25,56/MWh | -- |
| LightGBM (parámetros por defecto) | **19,75%** | **$13,12/MWh** | -- |
| XGBoost | 20,92% | $13,88/MWh | -- |
| MLP PyTorch — ReLU | 22,50% | $14,94/MWh | 0,52 |
| MLP PyTorch — GELU | 21,89% | $14,57/MWh | 0,53 |
| MLP PyTorch — Swish (elegida en validación) | 22,41% | $14,84/MWh | 0,55 |

**Con precios reales la MLP pierde contra ambos árboles y no le gana a la persistencia.** En el panel simulado era el mejor modelo (7,83% contra 10,07% de LightGBM por defecto); aquí las tres activaciones quedan en 21,9-22,5%, sobre LightGBM por defecto (19,75%), el ajustado (18,71%) y más o menos al nivel de la persistencia (21,88%). La activación elegida por la loss de validación (Swish) no es la mejor en test (GELU): las tres están a menos de 0,6 puntos entre sí, así que elegir entre ellas es ruido a esta escala.

**Protocolo de entrenamiento.** Dentro de cada fold walk-forward, el último 15% de las horas de entrenamiento se reserva como validación; las horas de test del fold no se tocan hasta evaluar. Cada red entrena al menos 100 épocas (máximo 300), luego el early stopping corta tras 20 épocas sin mejora en la loss de validación y restaura los pesos de la mejor época; la tasa de aprendizaje se reduce a la mitad en las mesetas. El estado completo se guarda en un checkpoint cada época (`data/checkpoints/`). **Las 15 corridas alcanzan su mejor loss de validación entre las épocas 5 y 27** y se detienen en el mínimo de 100: con precios reales la red empieza a sobreajustar casi de inmediato, y la regla del mínimo de épocas solo agrega épocas que el early stopping después descarta. Las métricas se guardan en `data/processed/metrics.duckdb` (tabla `model_comparison`).

![Curva de loss de la MLP en PyTorch, animada](data/processed/torch_loss_curve_marginal_cost_usd_mwh_animated.gif)
![Curva de loss de la MLP en PyTorch](data/processed/torch_loss_curve_marginal_cost_usd_mwh.png)

### 5.5 Intervalos de predicción: regresión cuantílica conformal que sigue a la estación (`src/models/conformal.py`)

Un par de regresores cuantílicos LightGBM (percentiles 5 y 95, un intervalo nominal de 90%) por target, conformalizados con Conformalized Quantile Regression (Romano, Patterson y Candès, 2019) sobre un tramo de calibración tomado del 15% más reciente del entrenamiento de cada fold walk-forward. Los hiperparámetros son los que Optuna ajustó para el modelo puntual; solo cambia la loss.

**Un margen fijo acierta en promedio y falla fold a fold.** Se calibra una vez con los últimos meses del entrenamiento y después se usa durante los meses siguientes, que son otra estación: la cobertura de la solar va de 73% a 94% según el fold, la de la demanda de 81% a 90%. `src/models/conformal_shift.py` compara cuatro formas de calibrar con los mismos modelos de cuantiles y folds:

| Calibración | Solar: media / peor fold | Eólica | Demanda | Precio | Ancho solar |
|---|---|---|---|---|---|
| Margen fijo, último 15% del entrenamiento | 84,1% / 72,7% | 89,5% / 87,1% | 87,9% / 81,4% | 88,7% / 85,3% | 546 MW |
| Mismas semanas del calendario de años anteriores | 80,2% / 69,5% | 87,2% / 84,4% | 84,8% / 77,4% | 91,2% / 86,0% | 1.248 MW |
| Móvil: recalibrar con las últimas horas observadas | 88,4% / 85,3% | 89,7% / 88,7% | 88,8% / 83,9% | 90,6% / 89,7% | 744 MW |
| **Adaptive Conformal Inference** | **89,7% / 89,2%** | **90,0% / 89,9%** | **89,9% / 89,7%** | **90,2% / 89,9%** | 840 MW |

**Adaptive Conformal Inference (Gibbs y Candès, 2021) mantiene todos los folds de todos los targets entre 89% y 91%.** Después de cada hora, cuando se conoce su valor real, mueve el nivel del cuantil en `gamma × (alpha − fallo)`: una racha de fallos ensancha los intervalos siguientes, una racha de horas fáciles los angosta. `gamma = 0,005` se fijó antes de mirar resultados. El costo es un ancho honesto: la banda de la solar crece de 546 a 840 MW, porque los mediodías de verano que no cubría de verdad son así de inciertos. Calibrar con las mismas semanas de años anteriores es peor en todos los targets: esas semanas salen del entrenamiento de los modelos de cuantiles, y la capacidad solar crece tan rápido que el verano del año pasado no es el de este año.

Los intervalos desplegados (`forecast.py`, el dashboard) usan el margen de ACI al final de los datos: para la solar, 1.197 MW a fines de diciembre de 2025, contra 112 MW del margen fijo calibrado entre octubre de 2024 y abril de 2025. En operación el nivel seguiría actualizándose a medida que llega el valor real de cada hora.

**Lo que ninguna de las cuatro arregla: la hora del día.** ACI corrige la cobertura en el tiempo, no dentro del día: la peor hora de la solar sigue cubierta solo el 66% de las veces (las horas de noche quedan sobrecubiertas, las rampas del amanecer y el atardecer subcubiertas). Una variante Mondrian, un margen por hora del día, solo mueve la peor hora de la solar de 55% a 60% y empeora la de la demanda y el precio (`conformal.py` la reporta junto a las demás); los intervalos desplegados no la usan.

**Cruce de cuantiles.** Los dos regresores cuantílicos se entrenan por separado, así que nada impide que se crucen en una fila; `rectify_crossing` aplica el arreglo estándar (mínimo/máximo elemento a elemento, Chernozhukov, Fernández-Val y Galichon, 2010) en todo lugar donde se construye o evalúa un intervalo, y `tests/test_forecast.py::test_forecast_intervals_are_never_crossed` lo fija sobre la salida recursiva de `forecast_node`.

Corre `python -m src.models.conformal` después de `train_forecaster.py` (reutiliza los hiperparámetros ajustados de `forecaster_metrics.json`), y `python -m src.models.conformal_shift` para la comparación de arriba.

## 6. Ejemplo de pronóstico multi-step

```bash
python -m src.models.forecast --horizon 12 --nodes Crucero_220kV,Puerto_Montt_220kV
```

Salida real, partiendo a medianoche local del 2026-01-01 (la primera hora después de los datos): el pronóstico del sistema pone la solar en 1,15 GW a las 07:00 y en 6,9 GW a las 11:00. Con esa trayectoria del sistema, y con los pesos de mezcla desplegados, **el precio de Crucero cae de 52-62 USD/MWh en la noche a 2 USD/MWh a las 08:00 y a 6-16 entre las 09:00 y las 11:00**, mientras **el de Puerto Montt se queda en 38-39 USD/MWh durante la mañana** (48-51 en la noche): el desacople norte-sur de §2. Con un peso del modelo de 0,25, el pronóstico de Puerto Montt arrastra sobre todo su último valor, que es lo que eligió la validación. La solar pronosticada de noche no es exactamente cero (19-73 MW, alrededor de 1% del máximo): el modelo no tiene una restricción física, solo un piso en 0.

**Limitación:** `forecast.py` persiste los últimos valores de clima *conocidos* hacia las horas futuras (ver `forecast_node`, `last_known_weather`), porque el proyecto usa observaciones meteorológicas históricas, no un pronóstico meteorológico. La solar sigue la hora del día porque la codificación cíclica se recalcula para cada hora futura, pero la eólica solo sigue sus propios rezagos. Un despliegue en producción necesitaría un pronóstico meteorológico real para el horizonte.

## 7. Stack tecnológico

Python · **PyMuPDF** (extracción de gráficos vectoriales de los informes diarios del CEN) · Polars · pandas · NumPy · LightGBM (puntual + regresión cuantílica) · XGBoost · **PyTorch** (MLP, loss Huber propia, ReLU/GELU/Swish) · **Conformalized Quantile Regression** · **DuckDB** (almacén de métricas) · Optuna · scikit-learn (`TimeSeriesSplit`) · Streamlit · Plotly · `holidays` · Jupyter/matplotlib (2 notebooks) · pytest (87 tests: geometría y lectura de la extracción de gráficos, validación cruzada de las series reconstruidas, estructura del panel y filas por target, fuga de variables, invariantes de los folds, baselines, pronóstico recursivo incluido el acople sistema-barra, cobertura y cruce conformal, MLP/loss Huber en PyTorch, ida y vuelta en DuckDB, smoke test del dashboard)

## 8. Cómo ejecutar

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Opcional: reconstruir las series horarias reales desde los PDF del CEN (alrededor de 1 GB,
# 1-2 horas a una consulta por segundo). Las series extraídas ya están versionadas en data/real/.
python -m src.data.cen_daily_reports download 2022-01-01 2025-12-31
python -m src.data.cen_daily_reports extract
python -m src.data.report_figures           # figuras de validación en data/processed/real_*.png

python -m src.data.fetch_energy_data        # arma el panel real (6 barras + SEN, clima)
python -m src.features.build_features       # rezagos, ventanas móviles, ciclos, clima
python -m src.models.train_forecaster       # tuning Optuna + walk-forward, los 4 targets
python -m src.models.rolling_stability      # Rolling-Origin CV, LightGBM vs XGBoost
python -m src.models.generate_torch_report  # MLP PyTorch vs LightGBM/XGBoost -> gráficos + metrics.duckdb
python -m src.models.conformal              # intervalos CQR (requiere train_forecaster antes)
python -m src.models.price_drivers          # qué arregla Puerto Montt: variables, modelos por barra, mezcla
python -m src.models.conformal_shift        # calibración de intervalos bajo cambio de estación

python -m src.models.forecast --horizon 24 --nodes all   # pronóstico recursivo (CLI)
streamlit run src/app/dashboard.py                        # dashboard de monitoreo

python -m pytest tests/ -v

jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_seasonality_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Weather_Augmented_Rolling_CV.ipynb
```

## 9. Estructura del proyecto

```
src/
  data/       cen_daily_reports.py    descarga los PDF diarios del CEN; lee de sus gráficos vectoriales
                                       el precio y la generación horarios, calibrados con el texto del informe
              real_panel.py           panel real: 6 barras + nodo SEN, limpieza, rows_for_target
              real_sources.py         clima Open-Meteo; series mensuales oficiales usadas para validar
              report_figures.py       figuras de reconstrucción vs. series oficiales y precio por barra
              fetch_energy_data.py    punto de entrada (panel real; simulador conservado como use_real=False)
  features/   build_features.py       rezagos, ventanas móviles, ciclos, calendario y clima con Polars
  models/     validation.py           folds walk-forward y Rolling-Origin, WAPE (global y por nodo)
              baselines.py            métricas de referencia ingenua / estacional
              train_forecaster.py     tuning Optuna + LightGBM, 4 targets; precio mezclado con la persistencia
              blend.py                peso de mezcla modelo/persistencia por barra, elegido dentro de cada fold
              price_drivers.py        experimento: variables extra, modelos por barra, target de cambio, mezcla
              rolling_stability.py    Rolling-Origin CV, LightGBM vs XGBoost
              torch_forecaster.py     MLP PyTorch, loss Huber, comparación ReLU/GELU/Swish
              torch_plots.py          gráficos de predicho vs. real, residuos, curva de loss y activaciones
              metrics_db.py           almacén de métricas DuckDB (tabla model_comparison)
              generate_torch_report.py  entrenamiento MLP + gráficos + persistencia en DuckDB
              conformal.py            intervalos de predicción CQR, 4 targets; margen adaptativo (ACI) desplegado
              conformal_shift.py      experimento: calibración fija, estacional, móvil y adaptativa
              forecast.py             pronóstico recursivo multi-step (primero el sistema, después las barras)
  app/        dashboard.py            monitoreo en Streamlit + pronóstico en vivo, con banda de intervalo
data/real/    cen_cmg_hourly.csv.gz, cen_generation_hourly_by_tech.csv.gz, cen_report_extraction_quality.csv,
              weather_hourly_open_meteo.csv.gz, sen_generacion_mensual.csv, cne_crucero_cmg_mensual.csv
notebooks/    01_eda_seasonality_analysis.ipynb       EDA del panel real: estacionalidad, precio por barra,
                                                       clima vs. generación, autocorrelación en los rezagos usados
              02_Weather_Augmented_Rolling_CV.ipynb   ablación de variables de clima + análisis de estabilidad
tests/                                pytest: extracción, fuentes reales, features, validación, baselines,
                                       pronóstico, conformal, estabilidad, PyTorch, DuckDB, dashboard
```

## 10. Autor

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE). Datos: informes diarios de operación del Coordinador Eléctrico Nacional (públicos), Open-Meteo (CC BY 4.0), datos.gob.cl y reportes mensuales de la CNE para validación.
