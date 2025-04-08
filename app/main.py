# Build FastAPI here

#### Load trained model ####
import torch
from transformers import BertTokenizer, BertModel

BASE_MODEL = "bert-base-chinese"
MODEL_STATE_PATH = "model/bert_multitask_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ASPECTS = 18
NUM_LABELS_PER_ASPECT = 4
BERT_OUTPUT_SIZE = 768
DROPOUT_RATE = 0.1

ASPECT_COLUMNS = [
    "Location#Transportation",
    "Location#Downtown",
    "Location#Easy_to_find",
    "Service#Queue",
    "Service#Hospitality",
    "Service#Parking",
    "Service#Timely",
    "Price#Level",
    "Price#Cost_effective",
    "Price#Discount",
    "Ambience#Decoration",
    "Ambience#Noise",
    "Ambience#Space",
    "Ambience#Sanitary",
    "Food#Portion",
    "Food#Taste",
    "Food#Appearance",
    "Food#Recommend",
]
SENTIMENT_MAP = {0: "not_mentioned", 1: "negative", 2: "neutral", 3: "positive"}


class MultiTaskBert(torch.nn.Module):
    """Multi-task BERT model for sentiment analysis."""

    def __init__(self, num_aspects, num_labels_per_aspect):
        super(MultiTaskBert, self).__init__()
        self.bert = BertModel.from_pretrained(BASE_MODEL)
        self.dropout = torch.nn.Dropout(DROPOUT_RATE)
        self.classifiers = torch.nn.ModuleList(
            [
                torch.nn.Linear(BERT_OUTPUT_SIZE, num_labels_per_aspect)
                for _ in range(num_aspects)
            ]
        )

    def forward(self, input_ids, attention_mask):
        """Forward pass for the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits


def load_model(path):
    """Load the trained model from the specified path."""
    model = MultiTaskBert(NUM_ASPECTS, NUM_LABELS_PER_ASPECT)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    model.to(DEVICE)
    model.eval()
    return model


def load_tokenizer(name):
    """Load the tokenizer for the specified model."""
    return BertTokenizer.from_pretrained(name)


def prepare_input(tokenizer, review):
    """Prepare the input for the model."""
    inputs = tokenizer(
        review,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    return input_ids, attention_mask


def predict(model, tokenizer, review):
    """Predict sentiments for the given review."""
    input_ids, attention_mask = prepare_input(tokenizer, review)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    predictions = [torch.argmax(logit, dim=1).cpu().numpy()[0] for logit in logits]
    return predictions


def print_predictions(review, aspect_columns, sentiment_map, predictions):
    """Print the predicted sentiments for each aspect."""
    print("Review:", review)
    print("\nPredicted Sentiments:")
    for aspect, pred in zip(aspect_columns, predictions):
        sentiment = sentiment_map[pred]
        print(f"{aspect}: {sentiment}")


def main():
    new_review = "食物很好吃，服务很慢，环境很吵"  # "The food is delicious, the service is slow, the environment is noisy"

    tokenizer = load_tokenizer(BASE_MODEL)
    model = load_model(MODEL_STATE_PATH)

    predictions = predict(model, tokenizer, new_review)
    print_predictions(new_review, ASPECT_COLUMNS, SENTIMENT_MAP, predictions)


if __name__ == "__main__":
    main()
