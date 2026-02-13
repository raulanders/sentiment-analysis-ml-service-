# 🧠 Sentiment Analysis ML Service (ES / PT)

Production-ready sentiment analysis microservice for Spanish and Portuguese.

This repository contains my Data Science and ML Engineering contribution within a full-stack architecture.

---

## 👨‍💻 My Technical Contribution

I was responsible for:

- Designing and implementing the complete ETL pipeline (ES & PT)
- Training and comparing classical ML models
- Training TF-IDF + Logistic Regression (ES / PT)
- Fine-tuning RoBERTa (PT)
- Model evaluation and metric comparison
- FastAPI inference microservice
- Docker containerization

---

# 🎯 Problem

Automatically classify customer feedback into:

- NEGATIVE  
- NEUTRAL  
- POSITIVE  

With emphasis on:
- Class robustness (especially NEUTRAL)
- Production feasibility
- Cost vs performance tradeoff

---

# ⚙ System Architecture


ETL → Training → Evaluation → Model Persistence → FastAPI → Docker



The API detects language automatically and routes inference to the appropriate model.

---

# 📊 Modeling Strategy

## Classical Models (Baseline)

- Decision Tree
- Naive Bayes
- TF-IDF + Logistic Regression (ES & PT)

### Best Classical Model (Spanish)

**TF-IDF + Logistic Regression**

- Accuracy: 0.734
- F1 Macro: 0.678
- Strong performance on NEGATIVE and POSITIVE
- Main limitation: NEUTRAL class ambiguity

---

## Transformers

### BETO (Spanish)

- Accuracy: 0.7849
- F1 Macro: 0.7255
- Strong contextual understanding
- Improved performance over classical models

### RoBERTa (Portuguese)

- Fine-tuned `xlm-roberta-base`
- Optimized for macro F1
- Better contextual generalization
- Selected as final PT production model

---

# 🔍 Key Finding

The main bottleneck in sentiment analysis is not extreme polarity detection,
but correct classification of NEUTRAL class.

Transformers significantly improve contextual robustness,
especially for ambiguous and mixed-sentiment samples.

---

# 🚀 FastAPI Inference Service

Production-ready REST API:

### Endpoints


GET / → Service status
GET /health → Health check
POST /predict → Sentiment prediction



### Features

- Single model load at startup
- Automatic language detection (ES/PT)
- Dynamic routing (LogReg or RoBERTa)
- Structured JSON response:
    - predicted class
    - probability
- Input validation (Pydantic)
- Docker compatible

---

# 📦 Model Artifacts

Due to GitHub file size limits, trained models are stored externally:

🔗 Google Drive:
https://drive.google.com/drive/u/0/folders/1E9DPet3ManYqEAx_veyYyl51pNZ8OXxp

After downloading, place inside:

- `model_roberta_pt`
- `modelo_beto_es`

After downloading, place them inside:

api/models/

---

# 🐳 Docker

Build:

```bash
docker build -t sentiment-api .
```

Run:
```
docker run -p 8000:8000 sentiment-api
```

🧪 Example Request

```
POST /predict
{
  "text": "El servicio fue excelente"
}
```
Response:

```
{
  "prevision": "POSITIVO",
  "probabilidad": 0.94
}
```
