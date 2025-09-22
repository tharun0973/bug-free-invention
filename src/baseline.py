import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and clean data
df = pd.read_csv("data/reply_classification_dataset.csv")
df.dropna(subset=["reply", "label"], inplace=True)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|mailto\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

df["text"] = df["reply"].apply(clean_text)
df["label"] = df["label"].str.strip().str.lower()

# Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_encoded"], test_size=0.2, random_state=42
)

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
print("Baseline F1 Score:", f1_score(y_test, y_pred, average="weighted"))
