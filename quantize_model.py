# quantize_model.py
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

SRC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_DIR  = Path("models/dashbrain-onnx")
OUT_DIR   = Path("models/dashbrain-int8-onnx")
ONNX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Exporting to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(SRC_MODEL, export=True)
model.save_pretrained(str(ONNX_DIR))
tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
tokenizer.save_pretrained(ONNX_DIR)

print("Applying INT8 quantization...")
quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)

quant_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

quantizer.quantize(quantization_config=quant_config, save_dir=str(OUT_DIR))
tokenizer.save_pretrained(OUT_DIR)

print(f"Done. Quantized model saved to {OUT_DIR}")