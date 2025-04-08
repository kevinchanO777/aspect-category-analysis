# Build FastAPI here


#### Load trained model ####
import torch
from transformers import BertTokenizer, BertModel

BASE_MODEL = "bert-base-chinese"
MODEL_STATE_PATH = "model/bert_multitask_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ASPECTS = 18
NUM_LABELS_PER_ASPECT = 4
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


# Define the MultiTaskBert class (same as training)
class MultiTaskBert(torch.nn.Module):
    def __init__(self, num_aspects, num_labels_per_aspect):
        super(MultiTaskBert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = torch.nn.Dropout(0.1)
        self.classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(768, num_labels_per_aspect) for _ in range(num_aspects)]
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits


def load_model(path):
    model = MultiTaskBert(
        num_aspects=NUM_ASPECTS, num_labels_per_aspect=NUM_LABELS_PER_ASPECT
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_tokenizer(name):
    tokenizer = BertTokenizer.from_pretrained(name)
    return tokenizer


def prepare_input(tokenizer, review):
    inputs = tokenizer(
        review,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",  # PyTorch tensor format
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    return input_ids, attention_mask


def predict(model, tokenizer, review):
    input_ids, attention_mask = prepare_input(tokenizer, review)

    # Run the model
    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # Get predictions for all 18 aspects

    # Interpret the results
    # Convert logits to predicted labels (0: not_mentioned, 1: negative, 2: neutral, 3: positive)
    predictions = [torch.argmax(logit, dim=1).cpu().numpy()[0] for logit in logits]

    return predictions


def print_predictions(review, aspect_columns, sentiment_map, predictions):
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
    print_predictions(
        new_review,
        aspect_columns=ASPECT_COLUMNS,
        sentiment_map=SENTIMENT_MAP,
        predictions=predictions,
    )


if __name__ == "__main__":
    main()
