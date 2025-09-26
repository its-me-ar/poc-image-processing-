from pathlib import Path
import argparse
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import onnx_export_from_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional quantization step uses onnxruntime.quantization
try:
	from onnxruntime.quantization import quantize_dynamic, QuantType
	HAS_ONNXRUNTIME_QUANT = True
except Exception:
	HAS_ONNXRUNTIME_QUANT = False


def export_and_optionally_quantize(model_dir: Path, onnx_dir: Path, quantize: bool = True):
	"""Export HF model to ONNX and optionally produce an int8 quantized copy.

	Outputs:
	  - <onnx_dir>/model.onnx  (exported)
	  - <onnx_dir>/model.int8.onnx  (quantized, if requested and available)
	"""
	model_dir = Path(model_dir)
	onnx_dir = Path(onnx_dir)
	onnx_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loading HF model from {model_dir}")
	hf_model = AutoModelForSequenceClassification.from_pretrained(model_dir)

	print(f"Exporting ONNX model to {onnx_dir}")
	onnx_export_from_model(model=hf_model, output=onnx_dir, task="sequence-classification")

	exported = onnx_dir / "model.onnx"
	print(f"ONNX model exported to {exported}")

	if quantize:
		if not HAS_ONNXRUNTIME_QUANT:
			print("onnxruntime.quantization not available in this environment. Skipping quantization.")
			return
		quantized_path = onnx_dir / "model.int8.onnx"
		print(f"Quantizing {exported} -> {quantized_path}")
		quantize_dynamic(str(exported), str(quantized_path), weight_type=QuantType.QInt8)
		print(f"Quantized model written to {quantized_path}")


def smoke_test(onnx_dir: Path, text: str = "draw a red circle"):
	"""Simple smoke test that loads tokenizer and ORT model from the onnx dir and prints a prediction."""
	onnx_dir = Path(onnx_dir)
	print(f"Running smoke test using models in {onnx_dir}")
	tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
	onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)
	inputs = tokenizer(text, return_tensors="pt")
	outputs = onnx_model(**inputs)
	predicted_idx = outputs.logits.argmax(dim=-1).item()
	print(f"Input: {text}\nPredicted label index: {predicted_idx}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-dir", default=Path(__file__).parent / "intent_model", help="Path to HF model folder")
	parser.add_argument("--onnx-dir", default=Path(__file__).parent / "onnx_model", help="Output ONNX folder")
	parser.add_argument("--no-quant", action="store_true", help="Disable quantization step")
	parser.add_argument("--smoke-test", action="store_true", help="Run a smoke test after export")
	args = parser.parse_args()

	export_and_optionally_quantize(args.model_dir, args.onnx_dir, quantize=not args.no_quant)
	if args.smoke_test:
		smoke_test(args.onnx_dir)


if __name__ == "__main__":
	main()
