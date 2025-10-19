## DQN Deep Q-Network para Trading con 3 acciones (buy/sell/hold)

Notebook educativo de Aprendizaje por Refuerzo para trading discreto con un agente DQN que decide entre buy, sell y hold. Incluye entorno propio, monitorización de entrenamiento y evaluación out-of-sample.

## Objetivos

- **Formular trading como MDP** (estado, acciones, recompensa con Proceso de Decisión de Márkov).
- **Entrenar un DQN** (red densa que aproxima Q) con política ε-greedy.
- **Reglas realistas**: límite de inventario, penalización por ventas inválidas y liquidación al final.
- **Trazabilidad**: monitorización de métricas y checkpoint único.

## Datos

- Dos CSV de S&P 500 (Close) se descargan automáticamente a `data/`:
  - `GSPC_2001-10.csv` (train)
  - `GSPC_2011.csv` (test)
- Transformación simple: `Close/1000` (reescala para estabilidad numérica).

## MDP del problema

- **Estado S_t**:
  - Ventana de retornos logarítmicos `window_size` (contexto temporal).
  - Ratio de inventario `len(inventory)/5` en [0,1] (exposición del agente).
- **Acciones**: 0=buy, 1=sell, 2=hold.
- **Recompensa**:
  - Venta: `precio_actual - precio_compra`.
  - Venta sin inventario: `-0.01` (regularización).
  - Fin de episodio: liquidación de todas las posiciones restantes.

Analogía: un boxeador (agente) con 3 golpes: punch derecho, punch izquierdo y gancho izquierdo. El entorno es el combate (oponente, reglas, ring y tiempo) y el estado resume estamina/guardia propias, distancia y guardia del rival, openings recientes y tiempo restante. Final de partida, cuando el round termina!!

## Arquitectura del agente

- **Red**: `Dense(10, relu) → Dense(3)` (Q-values para buy/sell/hold).
- **Pérdida/optimizador**: MSE + Adam (`lr=1e-3`).
- **Replay buffer**: `deque(maxlen=memsize)` para romper correlación temporal.
- **Actualización**: `r + γ·max_a' Q(s',a')` (backup de Bellman) con `train_on_batch`.
- **Política**: ε-greedy (`epsilon_ini`, `epsilon_decay`, `epsilon_min`).

## Entrenamiento y monitorización

- Recolección de experiencias y entrenamiento cada `training_ratio` pasos.
- Decaimiento de ε al final de cada episodio.
- Métricas por episodio: `profit_hist`, `eps_hist`, `invalid_sells`.
- Gráficas: Profit por episodio y Epsilon por episodio.
- Guardado: se persiste un único checkpoint como `models/last_model.keras`.

## Evaluación

- Carga del checkpoint único: `models/last_model.keras`.
- `is_eval=True` → greedy (sin exploración).
- Dataset de test: `GSPC_2011` (out-of-sample) y visualización de decisiones.

## Hiperparámetros (edítalos en el notebook)

- **Datos**: `stock_name`, `window_size`.
- **Episodios**: `episode_count` (el bucle usa `range(episode_count + 1)`).
- **DQN**: `gamma`, `epsilon_ini`, `epsilon_decay`, `epsilon_min`.
- **Entrenamiento**: `memsize`, `batch_size`, `training_ratio`.

## Cómo ejecutar

- **Colab**: “Runtime → Run all” (se crean `data/` y `models/`).
- **Local (Windows)**:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install keras numpy pandas matplotlib tqdm
# abre el notebook y ejecuta
```

## Snippets útiles

```python
# guardar último (ya implementado en el notebook)
agent.model.save("models/last_model.keras")

# evaluación (greedy)
last_model_path = "models/last_model.keras"
agent = Agent(model=last_model_path, is_eval=True)
```

## Reproducibilidad (opcional)

```python
import random, numpy as np, os
seed = 42
random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
```

## Estructura esperada

- `data/`: CSV de entrenamiento y test (se descargan al ejecutar).
- `models/`: `last_model.keras` tras entrenar.
- Notebook principal con secciones: datos → entorno → agente → entrenamiento → evaluación.

## Limitaciones y próximos pasos

- Sin costes de transacción ni slippage; features mínimas.
- No hay target network/Double DQN ni Prioritized Replay.
- Ajustar `epsilon_decay` según `episode_count` para equilibrar exploración.
- Extender con costes, target network y selección de mejor modelo (validación temporal).

## Aviso

Proyecto con fines educativos, no es consejo financiero.

## Licencia

La de Thomas Bayes !!!
