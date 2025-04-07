# Build FastAPI here


#### Load trained model ####
import torch
from transformers import BertTokenizer

# Step 1: Load the trained model and tokenizer
# Assuming you saved the model as "bert_multitask_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ASPECTS = 18  # Number of aspects from your dataset
NUM_LABELS_PER_ASPECT = 4  # "not_mentioned", "negative", "neutral", "positive"


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


# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Load the model
model = MultiTaskBert(
    num_aspects=NUM_ASPECTS, num_labels_per_aspect=NUM_LABELS_PER_ASPECT
)
model.load_state_dict(torch.load("bert_multitask_model.pth"))
model.to(DEVICE)
model.eval()  # Set to evaluation mode (no training)

# Step 2: Prepare the new Chinese review
new_review = "食物很好吃，服务很慢，环境很吵"  # "The food is delicious, the service is slow, the environment is noisy"

# Tokenize the review
inputs = tokenizer(
    new_review,
    add_special_tokens=True,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",  # PyTorch tensor format
)

# Move inputs to the device (GPU or CPU)
input_ids = inputs["input_ids"].to(DEVICE)
attention_mask = inputs["attention_mask"].to(DEVICE)

# Step 3: Run the model
with torch.no_grad():  # No training, just prediction
    logits = model(input_ids, attention_mask)  # Get predictions for all 18 aspects

# Step 4: Interpret the results
# Convert logits to predicted labels (0: not_mentioned, 1: negative, 2: neutral, 3: positive)
predictions = [torch.argmax(logit, dim=1).cpu().numpy()[0] for logit in logits]

# Define sentiment mapping
sentiment_map = {0: "not_mentioned", 1: "negative", 2: "neutral", 3: "positive"}

# Your 18 aspects from the notebook
aspect_columns = [
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

# Step 5: Display the predictions
print(f"Review: {new_review}")
print("\nPredicted Sentiments:")
for aspect, pred in zip(aspect_columns, predictions):
    sentiment = sentiment_map[pred]
    print(f"{aspect}: {sentiment}")
