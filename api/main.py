# ============================================================
# 1) Imports
# ============================================================

from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langdetect import detect_langs, LangDetectException
import torch
import torch.nn.functional as F
import os
import joblib
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
)

# ============================================================
# 2) Configuración global (DEVICE / MAX_LENGTH)
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256


# ============================================================
# 3) Pipeline 1: BETO (BERT) para español (es)
# ============================================================


class BETOPipeline:
    def __init__(self, model_dir: str):
        self.device = DEVICE

        # ----------------------------
        # 3.1 Cargar configuración
        # ----------------------------
        config_path = os.path.join(model_dir, "config.pkl")
        config = joblib.load(config_path)

        self.class_names = config.get("CLASS_NAMES", ["Negativo", "Neutro", "Positivo"])
        self.max_len = config.get("MAX_LEN", 200)

        # ----------------------------
        # 3.2 Cargar tokenizer local
        # ----------------------------
        tokenizer_dir = os.path.normpath(os.path.join(model_dir, "tokenizer"))
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )

        # ----------------------------
        # 3.3 Cargar pesos del modelo (state_dict)
        # ----------------------------
        checkpoint_path = os.path.join(model_dir, "model.pth")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )

        # ----------------------------
        # 3.4 Renombrar keys del checkpoint (compatibilidad)
        # ----------------------------
        new_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith("bert.bert."):
                new_key = key.replace("bert.bert.", "bert.", 1)
            elif key.startswith("bert.classifier"):
                new_key = key.replace("bert.classifier", "classifier", 1)
            else:
                new_key = key
            new_checkpoint[new_key] = value

        # ----------------------------
        # 3.5 Construir modelo base + cargar pesos
        # ----------------------------
        self.model = BertForSequenceClassification.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased", num_labels=len(self.class_names)
        )

        self.model.load_state_dict(new_checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # ----------------------------
        # 3.6 Map de labels
        # ----------------------------
        self.id2label = {i: name.upper() for i, name in enumerate(self.class_names)}

    def predict(self, texts):
        label, _ = self._infer(texts[0])
        return [label]

    def predict_proba(self, texts):
        _, probs = self._infer(texts[0])
        return [probs]

    def _infer(self, text: str):
        # ----------------------------
        # 3.7 Tokenización
        # ----------------------------
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # ----------------------------
        # 3.8 Inferencia + softmax
        # ----------------------------
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        pred_id = int(probs.argmax())
        label = self.id2label[pred_id]

        return label, probs


# ============================================================
# 4) Pipeline 2: RoBERTa para portugués (pt)
# ============================================================


class RobertaPipeline:
    def __init__(self, model_dir: str):
        # ----------------------------
        # 4.1 Cargar tokenizer y modelo HF
        # ----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self.model.to(DEVICE)
        self.model.eval()

        # ----------------------------
        # 4.2 Labels desde config del modelo
        # ----------------------------
        self.id2label = self.model.config.id2label

    def predict(self, texts):
        label, _ = self._infer(texts[0])
        return [label]

    def predict_proba(self, texts):
        _, probs = self._infer(texts[0])
        return [probs]

    def _infer(self, text):
        # ----------------------------
        # 4.3 Tokenización
        # ----------------------------
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # ----------------------------
        # 4.4 Inferencia + softmax
        # ----------------------------
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        pred_id = probs.argmax()
        label = self.id2label[pred_id]

        return label, probs

def resolve_model_path(folder: str) -> str:
    # Si estás en Docker (Linux container), los modelos viven en /app/models
    docker_path = os.path.join("/app/models", folder)
    if os.path.exists(docker_path):
        return docker_path

    # Si estás en local, usa FastAPI/models/<folder>
    return os.path.join(os.path.dirname(__file__), "models", folder)


# ============================================================
# 5) Carga de modelos (UNA sola vez al iniciar la API)
# ============================================================

pipeline_es = BETOPipeline(resolve_model_path("model_b_es"))  # BETO(BERT) ES
pipeline_pt = RobertaPipeline(resolve_model_path("model_pt"))  # RoBERTa PT

# ============================================================
# 6) Esquemas Pydantic (Entrada / Salida)
# ============================================================


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Texto a analizar")


class Prevision(str, Enum):
    POSITIVO = "POSITIVO"
    NEGATIVO = "NEGATIVO"
    NEUTRO = "NEUTRO"


class PredictResponse(BaseModel):
    prevision: Prevision
    probabilidad: float = Field(..., ge=0.0, le=1.0)


# ============================================================
# 7) Inicialización FastAPI
# ============================================================

app = FastAPI(
    title="Sentiment DS API",
    version="1.0.0",
    description="Microservicio DS para análisis de sentimiento (ES/PT).",
)


# ============================================================
# 8) Endpoints base
# ============================================================


@app.get("/")
def root():
    """Endpoint raíz para verificar que la API está en línea."""
    return {"message": "API funcionando"}


@app.get("/health")
def health():
    """Endpoint de salud para verificar disponibilidad."""
    return {"status": "OK"}


# ============================================================
# 9) Endpoint principal /predict
# ============================================================


@app.post("/predict", response_model=PredictResponse)
def predict(data: TextInput):
    """
    Endpoint principal de predicción.
    Recibe un texto, detecta el idioma (ES/PT) y retorna el sentimiento y la probabilidad.
    """
    prevision, score = analyze_sentiment(data.text)
    return {"prevision": prevision, "probabilidad": score}


# ============================================================
# 10) Lógica de inferencia (detección de idioma + selección modelo)
# ============================================================
def validar_texto_input(text: str) -> None:
    if text is None or not str(text).strip():
        raise HTTPException(status_code=400, detail="Texto vacío o inválido")


def analyze_sentiment(text: str):
    validar_texto_input(text)

    try:
        langs = detect_langs(text)
    except LangDetectException:
        raise HTTPException(
            status_code=400, detail="No se pudo detectar el idioma del texto"
        )

    language = langs[0].lang
    confidence_lang = langs[0].prob

    if confidence_lang < 0.60:
        raise HTTPException(
            status_code=400,
            detail="No se pudo determinar el idioma con suficiente confianza",
        )

    # Selección del pipeline según idioma
    if language == "es":
        pipeline = pipeline_es
    elif language == "pt":
        pipeline = pipeline_pt
    else:
        raise HTTPException(
            status_code=400,
            detail="Idioma no soportado. Solo se admite español (es) y portugués (pt).",
        )

    # Predicción
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]

    confidence = float(max(probabilities))
    prediction = prediction.upper()

    # Respuesta final (mismo contrato)
    if prediction == "POSITIVO":
        return Prevision.POSITIVO, confidence
    elif prediction == "NEGATIVO":
        return Prevision.NEGATIVO, confidence
    else:
        return Prevision.NEUTRO, confidence
