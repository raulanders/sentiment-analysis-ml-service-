# FastAPI python model

Microservicio **FastAPI (Python)** para exponer un modelo de **análisis de sentimiento** vía HTTP.  
Este proyecto forma parte del **Hackathon NoCountry - Proyecto 1: SentimentAPI**.

---

## 🎯 Objetivo

Recibir un texto y devolver:

- **prevision:** `POSITIVO | NEGATIVO | NEUTRO` (en MAYÚSCULAS)
- **probabilidad:** número entre `0` y `1`

---

## ✅ Contrato (DS ↔ BE)

### POST `/predict`

**Request**
```json
{ "text": "El servicio fue excelente" }
```

**Response**
```json
{ "prevision": "POSITIVO", "probabilidad": 0.93 }
```

---

### GET `/health`

**Response**
```json
{ "status": "OK" }
```

---

### GET `/`

**Response**
```json
{ "message": "API funcionando" }
```

---

## 🧠 Modelos implementados (ES / PT)

Este microservicio implementa un flujo real de inferencia con **modelos Transformers**, seleccionando automáticamente el modelo según el idioma detectado.

### 🇪🇸 Español (ES)
- **Modelo:** BETO (BERT para español)
- **Framework:** PyTorch + Transformers
- **Carga de artefactos:**
  - Configuración: `config.pkl` (Joblib)
  - Pesos: `model.pth` (state_dict)
  - Tokenizer: carpeta local `tokenizer/`

### 🇵🇹 Portugués (PT)
- **Modelo:** RoBERTa para portugués
- **Framework:** PyTorch + Transformers
- **Carga de artefactos:**
  - `AutoTokenizer` + `AutoModelForSequenceClassification` desde carpeta local del modelo

---

## 🌍 Detección de idioma

Antes de predecir, el servicio detecta el idioma usando `langdetect`.

- Idiomas soportados: `es`, `pt`
- Umbral mínimo de confianza: `0.60`
- Si el idioma no es soportado o la confianza es baja, retorna error HTTP 400.

---

## 🚀 Ejecutar en local

### Requisitos
- Python **3.11+** (recomendado 3.11 / 3.12)
- pip

---

### 1) Crear entorno virtual

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2) Instalar dependencias
```bash
pip install -r requirements.txt
```

---

### 3) Levantar servidor
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### 4) Probar

- Swagger: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

---

## 🐳 Ejecutar con Docker (cross-platform)

### Build
```bash
docker build -t sentiment-ds .
```

### Run
```bash
docker run --rm -p 8000:8000 sentiment-ds
```

Luego probá:

- http://localhost:8000/docs  
- http://localhost:8000/health  

---

## 🔌 Integración con Backend Java

El Backend Java debe llamar a:

- Base URL: `http://localhost:8000`
- Predict path: `/predict`
- Health path: `/health`

Ejemplo:

```http
POST http://localhost:8000/predict
Content-Type: application/json

{"text":"..."}
```

---

## 📌 Estructura del proyecto

- `main.py` → API FastAPI + endpoints (`/predict`, `/health`) + lógica completa de inferencia (ES/PT)
- `requirements.txt` → dependencias del proyecto
- `Dockerfile` → imagen Docker para correrlo en cualquier entorno
- `.dockerignore` → evita copiar archivos innecesarios al build
- `models/` → carpeta con modelos y artefactos necesarios

Ejemplo esperado:

```bash
models/
  model_b_es/
    config.pkl
    model.pth
    tokenizer/
      vocab.txt
      tokenizer_config.json
      ...
  model_pt/
    config.json
    pytorch_model.bin
    tokenizer.json
    vocab.json
    merges.txt
    ...
```

---

## ⚠️ Notas importantes

- El servicio carga los modelos **una sola vez** al iniciar (mejor performance).
- Rutas de modelos:
  - **Docker:** `/app/models/<folder>`
  - **Local:** `./models/<folder>`
- Recomendación de recursos:
  - **RAM:** 2GB+ (mínimo recomendado)
  - **CPU/GPU:** funciona en CPU, y usa GPU si está disponible

---

## ❗ Manejo de errores (HTTP 400)

El servicio valida entrada y condiciones mínimas antes de inferir.

Ejemplos de errores:

### Texto vacío o inválido
```json
{ "detail": "Texto vacío o inválido" }
```

### Idioma no detectado
```json
{ "detail": "No se pudo detectar el idioma del texto" }
```

### Confianza insuficiente
```json
{ "detail": "No se pudo determinar el idioma con suficiente confianza" }
```

### Idioma no soportado
```json
{ "detail": "Idioma no soportado. Solo se admite español (es) y portugués (pt)." }
```

---

## 🧪 Ejemplos de prueba rápidos

### Positivo
```json
{ "text": "El servicio fue excelente, volvería a comprar sin duda." }
```

### Negativo
```json
{ "text": "El producto llegó roto y el soporte no respondió." }
```

### Neutro
```json
{ "text": "El pedido llegó ayer en la tarde." }
```
