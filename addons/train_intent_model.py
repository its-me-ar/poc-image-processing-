import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

# Load dataset (path relative to this script)
here = Path(__file__).parent
data_path = here / "commands.json"
data = json.loads(data_path.read_text())

# Create Hugging Face Datasets
train_dataset = Dataset.from_list(data["train"])
val_dataset = Dataset.from_list(data["validation"])

# Label mapping
labels = list(set([ex["label"] for ex in data["train"] + data["validation"]]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Map string labels to integer IDs
def map_labels(example):
    example["label"] = label2id[example["label"]]
    return example

train_dataset = train_dataset.map(map_labels)
val_dataset = val_dataset.map(map_labels)

datasets = DatasetDict({"train": train_dataset, "validation": val_dataset})

# Tokenizer
model_id = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=32)

# Tokenize datasets
tokenized = datasets.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# Metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Training arguments (compatible with transformers 4.56.2)
training_args = TrainingArguments(
    output_dir="intent_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model + tokenizer
trainer.save_model("intent_model")
tokenizer.save_pretrained("intent_model")

print("Training complete. Model saved to intent_model/")
