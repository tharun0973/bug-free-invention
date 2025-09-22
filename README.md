# SvaraAI Reply Classification Pipeline

## Overview
This project classifies email replies into `positive`, `negative`, or `neutral` using both a baseline model and a fine-tuned transformer.

Both models performed exceptionally well on this dataset. The baseline Logistic Regression with TF-IDF achieved 99.53% accuracy and F1 score, while the fine-tuned DistilBERT reached 100%. Given the small dataset size and simplicity of the task, the baseline is surprisingly strong. However, for production use, I’d prefer DistilBERT due to its robustness on unseen language patterns and better generalization across domains.

## Setup
Create a virtual environment and install dependencies:
conda create -n svara-env python=3.10  
conda activate svara-env  
pip install -r requirements.txt

## Train Transformer Model
Run the training script to fine-tune DistilBERT:
python src/train.py

## Run Baseline Model
Evaluate the baseline TF-IDF + Logistic Regression model:
python src/baseline.py

## Start FastAPI Server
Launch the API server:
uvicorn app:app --reload  
Visit: http://127.0.0.1:8000

## Test Prediction
Send a sample request to the `/predict` endpoint:
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Looking forward to the demo!"}'

## Sample Output
{"label": "positive", "confidence": 1.0}

## Project Structure
svara-reply-classifier-v2/  
├── data/  
│   └── reply_classification_dataset.csv  
├── models/  
│   └── reply_classifier/  
│   └── label_encoder.pkl  
├── src/  
│   ├── train.py  
│   └── baseline.py  
├── app.py  
├── requirements.txt  
├── README.md  
├── answers.md
