import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import transformers
import torch


def preprocess_text(text):
    words = jieba.cut(text)
    return " ".join(words)


def convert_sentiment(score):
    if score == -2:
        return "not_mentioned"
    elif score == -1:
        return "negative"
    elif score == 0:
        return "neutral"
    else:  # score == 1
        return "positive"


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Define aspects, e.g. Food#Appearance, Service#Price, etc.
    aspect_columns = [col for col in df.columns if col not in ["id", "review", "star"]]
    y = df[aspect_columns]

    # Convert sentiment scores to categorical labels
    y = df[aspect_columns].astype("object")
    for col in y.columns:
        y.loc[:, col] = y[col].apply(convert_sentiment)

    # Data preprocessing
    df["processed_review"] = df["review"].apply(preprocess_text)

    return df["processed_review"], y, aspect_columns


train_path = "../data/train.csv"
dev_path = "../data/dev.csv"
test_path = "../data/test.csv"

X_train, y_train, aspect_columns = load_and_preprocess_data(train_path)
X_dev, y_dev, _ = load_and_preprocess_data(dev_path)
X_test, y_test, _ = load_and_preprocess_data(test_path)


######### Train BERT #########

from transformers import BertModel, BertTokenizer
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy as np
import joblib
from tqdm import tqdm  # Import tqdm for progress bar

# Ensure the model is in evaluation mode and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertModel.from_pretrained("bert-base-chinese").to(device)
model.eval()  # Set to evaluation mode since we're only extracting embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


# Step 1: Define a function to generate BERT embeddings with a progress bar
def get_bert_embeddings(texts, batch_size=32):
    embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    # Use tqdm to show a progress bar for the batches
    for i in tqdm(
        range(0, len(texts), batch_size),
        total=num_batches,
        desc="Generating BERT embeddings",
    ):
        batch_texts = texts[i : i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(
            batch_texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=512,  # BERT's max input length
            return_tensors="pt",  # Return PyTorch tensors
            add_special_tokens=True,
        )

        # Move inputs to the device (CPU/GPU)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get BERT embeddings (no gradient computation to save memory)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding (first token) as the review embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        embeddings.append(cls_embeddings)

    # Concatenate all batch embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


print("Generating BERT embeddings for dev data...")
X_dev_bert = get_bert_embeddings(X_dev, batch_size=32)
