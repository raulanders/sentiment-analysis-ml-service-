# 🧠 Sentiment Analysis ML Service (ES/PT)

Microservicio de análisis de sentimiento para español y portugués, desarrollado como componente de Data Science dentro de una arquitectura full-stack.

Este repositorio concentra mi contribución técnica en:

- Pipeline ETL
- Modelado clásico (TF-IDF + Regresión Logística)
- Fine-tuning de Transformer (RoBERTa)
- Implementación de API con FastAPI
- Dockerización para despliegue reproducible

---

## 🎯 Objetivo

Construir, comparar y desplegar modelos capaces de clasificar comentarios en:

- NEGATIVO  
- NEUTRO  
- POSITIVO  

Evaluando no solo desempeño (Accuracy, F1), sino también robustez semántica y viabilidad de producción.

---

# ⚙ Arquitectura del Proyecto

El proyecto cubre el ciclo completo: `ETL → Entrenamiento → Evaluación → Persistencia → API → Docker`


---

## 🗂 1. ETL (Extracción y Preparación de Datos)

Implementación de pipeline para:

- Limpieza y normalización de texto
- Eliminación de nulos y duplicados
- Etiquetado desde estrellas (1–2 negativo, 3 neutro, 4–5 positivo)
- Muestreo estratificado
- Consistencia entre datasets ES y PT

El objetivo fue generar datasets comparables y robustos para entrenamiento.

---

## 🤖 2. Modelado

### Baseline Clásico – TF-IDF + Regresión Logística

Resultados (Portugués):

- Accuracy: **0.872**
- F1 Macro: **0.780**
- F1 Weighted: **0.871**

Ventajas:

- Bajo costo computacional
- Inferencia rápida en CPU
- Alta escalabilidad

---

### Transformer – RoBERTa (xlm-roberta-base)

Resultados:

- Accuracy: **0.857**
- F1 Macro: **0.835**
- F1 Weighted: **0.858**

Hallazgo clave:

Aunque el accuracy global fue ligeramente menor, RoBERTa mejoró el F1 Macro, mostrando mejor balance entre clases y mayor robustez contextual, especialmente en la clase NEUTRO.

---

# 🚀 3. API – FastAPI

Implementé un microservicio en FastAPI que:

- Carga modelos una sola vez al iniciar la aplicación
- Detecta automáticamente el idioma (ES/PT)
- Enruta dinámicamente al modelo correspondiente
- Expone endpoints REST:

```
GET / → Estado básico
GET /health → Health check
POST /predict → Predicción de sentimiento
```

Incluye:

- Validación de entrada con Pydantic
- Manejo controlado de errores HTTP
- Contrato consistente de salida (clase + probabilidad)
- Compatibilidad local y en contenedor Docker

---

# 🐳 4. Docker

El servicio fue dockerizado para:

- Aislamiento de dependencias
- Entorno reproducible
- Portabilidad entre desarrollo y producción
- Integración sencilla con backend (Spring Boot)

---

# 📊 Enfoque de decisión técnica

Se realizó análisis costo-beneficio entre modelos clásicos y Transformers considerando:

- Desempeño (Accuracy, F1)
- Robustez en clase NEUTRO
- Latencia
- Escalabilidad
- Costo computacional (CPU vs GPU)

Decisión arquitectónica:

- RoBERTa como modelo principal cuando la prioridad es calidad.
- TF-IDF + Regresión Logística como fallback ligero y altamente escalable.

---

# 🏗 Stack Tecnológico

- Python
- Pandas
- NumPy
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- FastAPI
- Docker

---

# 📌 Contexto

Este repositorio corresponde a mi contribución técnica dentro de un proyecto full-stack desarrollado en equipo, donde fui responsable del pipeline de datos, modelado y despliegue del microservicio de inferencia.

