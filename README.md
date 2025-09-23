# SvaraAI Reply Classification Pipeline

## 🚀 Overview  
SvaraAI classifies email replies into \`positive\`, \`negative\`, or \`neutral\` using two models:  
- ✅ Baseline: TF-IDF + Logistic Regression (99.53% accuracy)  
- ✅ Transformer: Fine-tuned DistilBERT (100% accuracy)  

While the baseline performs well, the transformer generalizes better to unseen language patterns — making it ideal for production use.

---

## ⚙️ Setup  
Create a virtual environment and install dependencies:  
\`\`\`bash
conda create -n svara-env python=3.10  
conda activate svara-env  
pip install -r requirements.txt
\`\`\`

---

## 🧠 Train Transformer Model  
Run the training script to fine-tune DistilBERT:  
\`\`\`bash
python src/train.py  
\`\`\`

This script:  
- Cleans and encodes the dataset  
- Fine-tunes DistilBERT  
- Saves model, tokenizer, and label encoder to \`models/\`

---

## 🧪 Run Baseline Model  
Evaluate the baseline TF-IDF + Logistic Regression model:  
\`\`\`bash
python src/baseline.py
\`\`\`

---

## 🌐 Start FastAPI Server  
Launch the API server:  
\`\`\`bash
uvicorn app:app --reload  
\`\`\`  
Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI

---

## 🔍 Test Prediction  
Send a sample request to the \`/predict\` endpoint:  
\`\`\`bash
curl -X POST http://127.0.0.1:8000/predict \\
-H "Content-Type: application/json" \\
-d '{"text": "Looking forward to the demo!"}'
\`\`\`

---

## ✅ Sample Output  
\`\`\`json
{ "label": "positive", "confidence": 1.0 }
\`\`\`

---

## 🧪 Real-World Examples

| Input Text                                                  | Predicted Label | Confidence |
|-------------------------------------------------------------|------------------|------------|
| "Can we discuss pricing??"                                  | neutral          | 1.0        |
| "I'm excited to explore this further, plz send contract"    | positive         | 1.0        |
| "We not looking for new solutions."                         | negative         | 1.0        |

---

## 🐳 Docker Deployment

To run the app inside a container:

### 1️⃣ Build the image
\`\`\`bash
docker build -t svara-api .
\`\`\`

### 2️⃣ Run the container
\`\`\`bash
docker run -p 8000:8000 svara-api
\`\`\`

### 3️⃣ Access the API
Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

This setup ensures reproducibility across machines and simplifies deployment to cloud platforms.

---

## 📁 Project Structure

\`\`\`
svara-reply-classifier-v2/  
├── data/  
│   └── reply_classification_dataset.csv  
├── models/  
│   ├── reply_classifier/  
│   └── label_encoder.pkl  
├── src/  
│   ├── train.py  
│   └── baseline.py  
├── app.py  
├── requirements.txt  
├── Dockerfile  
├── README.md  
├── answers.md  
\`\`\`

---

## 🏁 Bonus Implementation

✅ Added \`requirements.txt\` for reproducible local setup  
✅ Added \`Dockerfile\` for containerized deployment and portability  
✅ Verified predictions via \`curl\` and Swagger UI inside Docker

---

## 👨‍💻 Author

Built by [Tharun Kumar](https://github.com/tharun0973)  
AI Engineer focused on scalable document analysis bots, ML deployment, and enterprise-ready workflows.
