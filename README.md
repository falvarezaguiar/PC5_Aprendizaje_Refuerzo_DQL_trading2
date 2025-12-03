## DQN para Trading con 3 acciones (buy/sell/hold)

Notebook educativo de Aprendizaje por Refuerzo para trading discreto con un agente DQN que decide entre buy, sell y hold. Incluye entorno propio, monitorización de entrenamiento y evaluación out-of-sample.

## Objetivos

- **Formular trading como MDP** (estado, acciones, recompensa).
- **Entrenar un DQN** (red densa que aproxima Q) con política ε-greedy.
- **Reglas realistas**: límite de inventario, penalización por ventas inválidas y liquidación al final, Market-to-Market (M2M) reward continuo según sea el caso
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

Analogía: un boxeador (agente) con 3 golpes: punch derecho, punch izquierdo y gancho izquierdo. El entorno es el combate (oponente, reglas, ring y tiempo) y el estado resume estamina/guardia propias, distancia y guardia del rival, openings recientes y tiempo restante.

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
- **Entrenamiento incremental por tiempo**: límite configurable (ej. 4 horas).
- **Checkpoints automáticos**: cada 10 episodios y al finalizar.
- Guardado: checkpoints completos en `checkpoints/` y modelo final en `models/`.

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

- **Colab**: "Runtime → Run all" (se crean `data/`, `models/` y `checkpoints/`).
- **Local (Windows)**:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install keras numpy pandas matplotlib tqdm tensorflow
# abre el notebook y ejecuta
```

## Entrenamiento incremental

El notebook incluye un sistema de checkpoints para entrenar por tiempo limitado y continuar después:

### Entrenar desde cero (primera vez)

1. En la celda de configuración, establece: `CONTINUE_TRAINING = False`
2. Ajusta `TRAINING_TIME_HOURS = 4` (o el tiempo que desees)
3. Ejecuta todas las celdas
4. El entrenamiento se detendrá después del tiempo límite
5. Se guardará un checkpoint en `checkpoints/checkpoint_dqn_trading.pkl`

### Continuar entrenamiento previo

1. Establece: `CONTINUE_TRAINING = True`
2. Ejecuta todas las celdas
3. El sistema cargará automáticamente el último checkpoint
4. Continuará entrenando desde donde lo dejaste

### Detalles de los checkpoints

- **Frecuencia**: cada 10 episodios + checkpoint final
- **Contenido**: pesos del modelo, epsilon, últimas 500 experiencias, métricas
- **Ubicación**: `checkpoints/checkpoint_dqn_trading.pkl`
- **Modelo final**: `models/model_ep_final.keras`

### Configuración GPU

El notebook detecta automáticamente GPUs disponibles y las configura para:

- Crecimiento dinámico de memoria (evita reservar toda la GPU)
- Funcionamiento correcto en CPU si no hay GPU (solo más lento)

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
- `models/`: Modelos Keras guardados (`model_ep_final.keras`).
- `checkpoints/`: Checkpoints completos para continuar entrenamiento (`.pkl`).
- Notebook principal con secciones: datos → entorno → agente → entrenamiento → evaluación.

## Limitaciones y próximos pasos

- Sin costes de transacción ni slippage; features mínimas.
- No hay target network/Double DQN ni Prioritized Replay.
- Ajustar `epsilon_decay` según `episode_count` para equilibrar exploración.
- Extender con costes, target network y selección de mejor modelo (validación temporal).

## Notebooks Incluidos

### **PC5_DQN_Trading_Basico.ipynb**

**Arquitectura:**

- Dense(**8**, relu) → Dense(3)
- 11 features: 10 rendimientos logarítmicos + 1 ratio inventario
- Loss: MSE, Optimizer: Adam(lr=0.001)

**Hyperparámetros:**

- Episodios: **200**
- Epsilon: inicial=1.0, mínimo=0.05, decay=**0.99**
- Gamma: 0.95
- Memoria: 1000, Batch: 32, Training ratio: 64
- **Comisiones**: 0.2% por transacción

**Sistema de Guardado:**

- ✅ Guarda cada **10 episodios** en Google Drive
- ✅ Formato: `model_ep000.keras`, `model_ep010.keras`, ..., `model_ep200.keras`
- ⚠️ Solo pesos del modelo (no estado completo: memoria/epsilon)

---

### **PC5_DQN_Trading_Intermedio.ipynb**

**Arquitectura:**

- Dense(**10**, relu) → Dense(3)
- 11 features (igual que Básico)
- Loss: MSE, Optimizer: Adam(lr=0.001)

**Hyperparámetros:**

- Episodios: **100**
- Epsilon: inicial=1.0, mínimo=0.01, decay=**0.95**
- Gamma: 0.95
- Memoria: 1000
- **Sin comisiones**

**Sistema de Guardado:**

- ✅ Guarda solo al final: `models/last_model.keras`
- ❌ Sin guardado incremental

---

## ⚠️ Limitaciones de Ambos Notebooks

**NO implementan:**

- ❌ Checkpoints completos (`.pkl` con memoria/epsilon/métricas)
- ❌ Variable `CONTINUE_TRAINING` para reanudar entrenamiento
- ❌ Sistema de entrenamiento incremental por tiempo

**Para entrenar:**

- Ejecutar sesión completa de principio a fin
- **Básico**: ~200 episodios, recuperable cada 10 ep (solo pesos)
- **Intermedio**: ~100 episodios, solo modelo final

## Tabla Comparativa: Básico vs Intermedio

| Característica            | **Básico**    | **Intermedio**  |
| ------------------------- | ------------- | --------------- |
| **Neuronas ocultas**      | 8             | 10              |
| **Episodios**             | 200           | 100             |
| **Epsilon decay**         | 0.99 (suave)  | 0.95 (agresivo) |
| **Epsilon mínimo**        | 0.05          | 0.01            |
| **Comisiones**            | ✅ 0.2%       | ❌ No           |
| **Guardado incremental**  | ✅ Cada 10 ep | ❌ Solo final   |
| **Ubicación modelos**     | Google Drive  | Local `models/` |
| **Tiempo estimado**       | ~2-3 horas    | ~1-1.5 horas    |
| **Checkpoints completos** | ❌            | ❌              |
| **CONTINUE_TRAINING**     | ❌            | ❌              |

**Recomendación:**

- **Básico**: Mejor para aprendizaje completo (más episodios, exploración gradual, guardado incremental)
- **Intermedio**: Mejor para experimentación rápida (menos episodios, red más grande)

## Aviso

Proyecto con fines educativos, no es consejo financiero.

## Licencia

Indica una licencia (p.ej., MIT) en este archivo si lo publicas.
