from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import json

# Paths
onnx_path = Path("onnx_model")
config_path = onnx_path / "config.json"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(onnx_path)

# Load ONNX model
onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_path)

# Load label mapping from config
with open(config_path) as f:
    config = json.load(f)
id2label = config.get("id2label", {})

def predict_label(text: str) -> str:
    """
    Predicts the label for a given text using the ONNX model.
    Returns the label name.
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")  # or "np" for numpy

    # Run inference
    outputs = onnx_model(**inputs)
    
    # Get predicted label index
    predicted_idx = outputs.logits.argmax(dim=-1).item()
    
    # Map to label name
    return id2label.get(str(predicted_idx), str(predicted_idx))

# Example usage
text_input = "draw a red circle"
predicted_label = predict_label(text_input)
print(f"Input: {text_input}\nPredicted label: {predicted_label}")
