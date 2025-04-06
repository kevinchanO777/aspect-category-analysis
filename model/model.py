import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import jieba
import joblib

# Load data from CSV
data_path = "../data/test.csv"
df = pd.read_csv(data_path)

# Define aspect categories
aspect_columns = [col for col in df.columns if col not in ["id", "review", "star"]]


# Text preprocessing with Chinese segmentation
def preprocess_text(text):
    words = jieba.cut(text)
    return " ".join(words)


# Prepare features
df["processed_review"] = df["review"].apply(preprocess_text)

# Inspect processed_review
print("\nProcessed reviews (first 3 samples):")
for i in range(min(3, len(df))):
    print(f"\nRow {i}:")
    print(f"Original: {df['review'].iloc[i]}")
    print(f"Processed: {df['processed_review'].iloc[i]}")

X = df["processed_review"]

# Prepare target variables - Keep as DataFrame
y = df[aspect_columns].astype("object")


# Convert sentiment scores to categorical labels
def convert_sentiment(score):
    if score == -2:
        return "not_mentioned"
    elif score == -1:
        return "negative"
    elif score == 0:
        return "neutral"
    elif score == 1:
        return "positive"


# Apply conversion to each column with .loc
for col in y.columns:
    y.loc[:, col] = y[col].apply(convert_sentiment)

# Text vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Verify shapes
print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Train model for each aspect with class checking
classifiers = {}
for aspect in aspect_columns:
    unique_classes = y_train[aspect].nunique()
    if unique_classes < 2:
        print(
            f"Skipping {aspect}: only one class ({y_train[aspect].iloc[0]}) in training data"
        )
        classifiers[aspect] = None
    else:
        clf = LinearSVC(random_state=42)
        clf.fit(X_train, y_train[aspect])
        classifiers[aspect] = clf

# Evaluation
print("\nModel Evaluation:")
for aspect in aspect_columns:
    if classifiers[aspect] is not None:
        y_pred = classifiers[aspect].predict(X_test)
        y_true = y_test[aspect]
        print(f"\n{aspect}:")
        print(classification_report(y_true, y_pred))
    else:
        print(f"\n{aspect}: No model trained (single class in training data)")


# Function to predict sentiments for new review
def predict_sentiments(review_text):
    processed_text = preprocess_text(review_text)
    text_tfidf = tfidf.transform([processed_text])

    predictions = {}
    for aspect, clf in classifiers.items():
        if clf is None:
            predictions[aspect] = "not_mentioned"  # Default for single-class aspects
        else:
            pred = clf.predict(text_tfidf)[0]
            predictions[aspect] = pred

    return predictions


# Example prediction
new_review = (
    "这家餐厅环境很好，服务态度也不错，菜品味道很棒，就是价格有点贵，停车不太方便。"
)
predictions = predict_sentiments(new_review)
print("\nPredictions for new review:")
for aspect, sentiment in predictions.items():
    print(f"{aspect}: {sentiment}")

# # Save the model
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
# for aspect, clf in classifiers.items():
#     if clf is not None:
#         joblib.dump(clf, f'classifier_{aspect}.pkl')
