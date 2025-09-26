from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import onnx_export_from_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths
here = Path(__file__).parent
model_path = here / "intent_model"
onnx_path = here / "onnx_model"  # MUST be a Path object

# Export model using onnx_export_from_model (supports task)
hf_model = AutoModelForSequenceClassification.from_pretrained(model_path)
onnx_export_from_model(model=hf_model, output=onnx_path, task="sequence-classification")
print(f"ONNX model exported to {onnx_path}")

# Test ONNX inference
tokenizer = AutoTokenizer.from_pretrained(model_path)
onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_path)

inputs = tokenizer("brighten image by 20%", return_tensors="pt")
outputs = onnx_model(**inputs)
predicted_label = outputs.logits.argmax(dim=-1).item()
print("Predicted label index:", predicted_label)
