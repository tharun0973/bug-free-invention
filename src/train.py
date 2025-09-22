import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Load and clean data
df = pd.read_csv("data/reply_classification_dataset.csv")
df.rename(columns={"reply": "text"}, inplace=True)
df.dropna(subset=["text", "label"], inplace=True)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|mailto\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)
df["label"] = df["label"].str.strip().str.lower()

# Encode labels and rename to 'labels' for Trainer compatibility
le = LabelEncoder()
df["labels"] = le.fit_transform(df["label"])

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df[["clean_text", "labels"]])

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["clean_text"], padding=True, truncation=True)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset = encoded_dataset.train_test_split(test_size=0.2)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# Training arguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="logs",
    logging_steps=10,
    disable_tqdm=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"]
)

# Train
trainer.train()
from sklearn.metrics import accuracy_score, f1_score

preds = trainer.predict(encoded_dataset["test"])
y_pred = preds.predictions.argmax(axis=1)
y_true = preds.label_ids

print("Transformer Accuracy:", accuracy_score(y_true, y_pred))
print("Transformer F1 Score:", f1_score(y_true, y_pred, average="weighted"))
  
import pickle
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

model.save_pretrained("models/reply_classifier")
tokenizer.save_pretrained("models/reply_classifier")
