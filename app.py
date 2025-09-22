from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy as np
import pickle

app = FastAPI()

# Load model and tokenizer
model_path = "models/reply_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

class InputText(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Reply classifier is running."}

@app.post("/predict")
def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = round(probs[0][pred_id].item(), 3)
        label = le.inverse_transform([pred_id])[0]
    return {"label": label, "confidence": confidence}
