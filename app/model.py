import torch
from transformers import BertTokenizer, BertModel


class MultiTaskSentimentModel:
    """A class encapsulating the custom multi-task BERT model for sentiment analysis."""

    # Default configuration
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

    def __init__(
        self,
        base_model=BASE_MODEL,
        model_state_path=MODEL_STATE_PATH,
        num_aspects=NUM_ASPECTS,
        num_labels_per_aspect=NUM_LABELS_PER_ASPECT,
        dropout_rate=DROPOUT_RATE,
    ):
        """Initialize the multi-task sentiment model."""
        self.base_model = base_model
        self.model_state_path = model_state_path
        self.num_aspects = num_aspects
        self.num_labels_per_aspect = num_labels_per_aspect
        self.dropout_rate = dropout_rate

        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """Load the trained model from the specified path."""
        model = MultiTaskBert(self.num_aspects, self.num_labels_per_aspect)
        try:
            model.load_state_dict(
                torch.load(self.model_state_path, map_location=self.DEVICE)
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        model.to(self.DEVICE)
        return model

    def _load_tokenizer(self):
        """Load the tokenizer for the specified model."""
        return BertTokenizer.from_pretrained(self.base_model)

    def _prepare_input(self, review):
        """Prepare the input for the model."""
        inputs = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.DEVICE)
        attention_mask = inputs["attention_mask"].to(self.DEVICE)
        return input_ids, attention_mask

    def predict(self, review):
        """Predict sentiments for the given review."""
        input_ids, attention_mask = self._prepare_input(review)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
        predictions = [torch.argmax(logit, dim=1).cpu().numpy()[0] for logit in logits]
        return predictions

    def print_predictions(self, review):
        """Print the predicted sentiments for each aspect."""
        predictions = self.predict(review)
        print("Review:", review)
        print("\nPredicted Sentiments:")
        for aspect, pred in zip(self.ASPECT_COLUMNS, predictions):
            sentiment = self.SENTIMENT_MAP[pred]
            print(f"{aspect}: {sentiment}")

    def get_sentiment_results(self, review):
        """Return sentiment predictions as a dictionary."""
        predictions = self.predict(review)
        return {
            aspect: self.SENTIMENT_MAP[pred]
            for aspect, pred in zip(self.ASPECT_COLUMNS, predictions)
        }


class MultiTaskBert(torch.nn.Module):
    """Multi-task BERT model for sentiment analysis (Same as the training one)."""

    def __init__(
        self,
        num_aspects,
        num_labels_per_aspect,
        base_model="bert-base-chinese",
        dropout_rate=0.1,
    ):
        super(MultiTaskBert, self).__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(768, num_labels_per_aspect) for _ in range(num_aspects)]
        )

    def forward(self, input_ids, attention_mask):
        """Forward pass for the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits


# For testing
if __name__ == "__main__":
    sentiment_model = MultiTaskSentimentModel()
    review = "餐厅服务很好，但食物有点贵"
    sentiment_model.print_predictions(review)
