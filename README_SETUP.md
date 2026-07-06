# Guía de instalación y ejecución — Replicación multi-seed

Este paquete corre la rejilla completa de **4 seeds × 3 densidades de datos (50%/25%/10%) = 12 corridas**,
cada una entrenando 25,000 pasos y extrayendo la trayectoria espectral continua (λmax + traza de Hutchinson).

---

## 1. Requisitos previos

### 1.1 Python
Necesitas **Python 3.9, 3.10, 3.11 o 3.12**. Verifica qué tienes instalado:

```bash
python3 --version
```

Si no tienes Python o tienes una versión muy vieja (<3.9):
- **Windows**: descarga desde https://www.python.org/downloads/ (marca "Add Python to PATH" durante la instalación).
- **macOS**: `brew install python@3.11` (requiere Homebrew: https://brew.sh)
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install python3 python3-venv python3-pip`

### 1.2 ¿Tienes GPU NVIDIA?
Esto determina qué versión de PyTorch instalar. Verifica:

```bash
nvidia-smi
```

- Si el comando **funciona** y muestra una tabla con tu GPU → tienes GPU NVIDIA disponible, sigue la instalación con CUDA (más rápido).
- Si dice "comando no encontrado" o no reconoce el hardware → vas a correr en CPU (más lento, pero **totalmente viable** para este experimento — ver estimaciones de tiempo abajo).

---

## 2. Crear el entorno virtual (recomendado, evita conflictos con otros proyectos)

```bash
# Dentro de la carpeta donde tengas estos scripts
python3 -m venv venv

# Activar el entorno:
# En Linux/macOS:
source venv/bin/activate
# En Windows (cmd):
venv\Scripts\activate.bat
# En Windows (PowerShell):
venv\Scripts\Activate.ps1
```

Sabrás que está activo porque tu terminal mostrará `(venv)` al inicio de la línea.

---

## 3. Instalar PyTorch

**Opción A — Tienes GPU NVIDIA (recomendado si la tienes, es ~10-20x más rápido):**

Primero verifica qué versión de CUDA soporta tu driver (aparece en la esquina superior derecha de la tabla de `nvidia-smi`, campo "CUDA Version"). Luego instala la build correspondiente. Para CUDA 12.x (lo más común en 2025-2026):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Si tu CUDA es distinto, usa el selector oficial para obtener el comando exacto:
https://pytorch.org/get-started/locally/

**Opción B — Solo CPU (sin GPU o no quieres lidiar con drivers CUDA):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Verifica que quedó instalado correctamente:**

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())"
```

Debe imprimir la versión y `True` o `False` según tu hardware, sin errores.

---

## 4. Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

(Esto instala `numpy`; PyTorch ya lo instalaste en el paso anterior con el comando específico de tu hardware).

---

## 5. Verificar que los archivos están todos en la misma carpeta

```
tu_carpeta/
├── generate_dataset.py
├── model_architecture.py
├── train_and_grok.py
├── hessian_topology.py
├── run_all_experiments.py
├── visualize_paper.py
├── requirements.txt
└── README_SETUP.md   (este archivo)
```

**No cambies los nombres ni la ubicación relativa de los archivos** — `train_and_grok.py` y `hessian_topology.py` importan directamente desde `generate_dataset.py` y `model_architecture.py`.

---

## 6. Espacio en disco necesario

Cada corrida guarda ~66 checkpoints del modelo (unos 1.7 MB cada uno, modelo de 422K parámetros) →
~110 MB por corrida × 12 corridas ≈ **1.3 GB en total**. Verifica que tengas al menos 2 GB libres.

---

## 7. Ejecutar

### 7.1 Primero, una verificación rápida (recomendado, toma ~1 minuto)

Antes de lanzar las 12 horas... perdón, las 12 corridas completas, valida que todo funciona con una corrida diminuta:

```bash
python3 train_and_grok.py --seed 99 --train_ratio 0.5 --max_steps 300 --run_name smoketest
python3 hessian_topology.py --run_name smoketest --train_ratio 0.5 --seed 99 --trace_samples 5
```

Si ambos terminan sin errores e imprimen números de `lambda_max` y `tr(H)`, estás listo. Limpia después:

```bash
# Linux/macOS
rm -rf checkpoints/smoketest grokking_telemetry_smoketest*.json
# Windows (PowerShell)
Remove-Item -Recurse -Force checkpoints\smoketest, grokking_telemetry_smoketest*.json
```

### 7.2 Ver el plan completo sin ejecutar nada (dry run)

```bash
python3 run_all_experiments.py --dry_run
```

Esto imprime las 12 combinaciones que se van a correr, sin gastar cómputo.

### 7.3 Lanzar la rejilla completa

```bash
python3 run_all_experiments.py
```

Esto corre las 12 combinaciones **de forma secuencial y automática**: entrena, luego extrae el espectro,
y pasa a la siguiente. Cada corrida genera su propio log en `logs/train_<run>.log` y `logs/hessian_<run>.log`,
y su propia telemetría en `grokking_telemetry_<run_name>_with_hessian.json`.

### 7.4 Si se corta a la mitad (se cierra la terminal, se va la luz, etc.)

**No pasa nada.** Vuelve a correr exactamente el mismo comando:

```bash
python3 run_all_experiments.py
```

El script detecta automáticamente qué corridas ya terminaron (revisa si existe su archivo
`grokking_telemetry_<run_name>_with_hessian.json` con datos válidos) y **salta las que ya están hechas**,
retomando solo lo pendiente.

### 7.5 Dejarlo corriendo en segundo plano (opcional, útil si vas a cerrar la terminal)

**Linux/macOS:**
```bash
nohup python3 run_all_experiments.py > run_all.log 2>&1 &
```
Puedes cerrar la terminal y seguirá corriendo. Para ver el progreso:
```bash
tail -f run_all.log
```

**Windows (PowerShell):** abre una ventana y déjala minimizada, o usa el Programador de Tareas si necesitas que sobreviva a un cierre de sesión. La forma más simple es simplemente dejar la ventana de PowerShell abierta y minimizada.

---

## 8. Tiempos estimados

Medido en un entorno CPU-only (sin GPU) equivalente a un procesador de escritorio moderno:

| Fase | Tiempo por corrida | Tiempo total (12 corridas) |
|---|---|---|
| Entrenamiento (25,000 pasos) | ~35 min | ~7 horas |
| Extracción espectral (66 checkpoints, λmax + traza) | ~7 min | ~1.4 horas |
| **Total** | **~42 min** | **~8.5 horas** |

Con GPU NVIDIA (incluso una modesta, tipo laptop con RTX serie 30/40), espera una reducción de
**5-10x** en el tiempo de entrenamiento (la extracción espectral con HVP se beneficia menos porque
ya está limitada por el número de backprops dobles, no por el tamaño del batch).

**Recomendación práctica:** lánzalo en la noche o mientras haces otra cosa; no requiere que estés
frente a la pantalla. Con `nohup` (Linux/macOS) puedes cerrar la terminal sin problema.

---

## 9. ¿Qué archivos me interesan al final?

Por cada una de las 12 corridas:

- `grokking_telemetry_<run_name>_with_hessian.json` → contiene `steps`, `train_loss`, `val_loss`,
  `val_accuracy`, `checkpoint_steps`, `lambda_max`, `hessian_trace`, y metadata (`seed`, `train_ratio`, `weight_decay`).

Estos 12 archivos son los que necesito para la Fase 4 (agregación estadística: media ± desviación
estándar, regeneración de figuras con bandas de confianza). Cuando termine la rejilla, súbeme esos
12 archivos JSON y seguimos con el análisis.

---

## 10. Problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | El entorno virtual no está activado, o la instalación falló | Verifica `(venv)` en el prompt; reinstala con el comando del paso 3 |
| `CUDA out of memory` | Muy improbable con este modelo (422K params), pero si pasa | Añade `--max_steps` más bajo para probar, o fuerza CPU con la variable de entorno `CUDA_VISIBLE_DEVICES=""` antes del comando |
| El entrenamiento es mucho más lento de lo estimado | Corriendo en CPU con pocos núcleos, o hay otros procesos pesados abiertos | Verifica con el Administrador de Tareas/`htop`; cierra programas innecesarios |
| `FileNotFoundError` al correr `hessian_topology.py` | No corriste `train_and_grok.py` primero para ese mismo `--run_name` | Corre primero el entrenamiento; `run_all_experiments.py` ya hace esto en el orden correcto automáticamente |
