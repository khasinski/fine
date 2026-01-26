# Exporting Models for Deployment

Fine supports exporting fine-tuned models to ONNX and GGUF formats for production deployment.

## ONNX Export

ONNX (Open Neural Network Exchange) is a cross-platform format supported by many inference runtimes including ONNX Runtime, TensorRT, and OpenVINO.

### Export Text Classifier

```ruby
classifier = Fine::TextClassifier.load("my_classifier")
classifier.export_onnx("classifier.onnx")

# With options
classifier.export_onnx(
  "classifier.onnx",
  opset_version: 14,
  dynamic_axes: true  # Allow variable batch size and sequence length
)
```

### Export Text Embedder

```ruby
embedder = Fine::TextEmbedder.load("my_embedder")
embedder.export_onnx("embedder.onnx")
```

### Export Image Classifier

```ruby
classifier = Fine::ImageClassifier.load("my_classifier")
classifier.export_onnx("classifier.onnx")
```

### Export LLM

```ruby
llm = Fine::LLM.load("my_llm")
llm.export_onnx("llm.onnx")
```

### Using the Export Module Directly

```ruby
Fine::Export.to_onnx(model, "model.onnx", opset_version: 14)
```

### ONNX Inference Example (Python)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("classifier.onnx")

# Text classification
input_ids = np.array([[101, 2054, 2003, 2023, 102]], dtype=np.int64)
attention_mask = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)

outputs = session.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask
})
logits = outputs[0]
```

## GGUF Export (LLMs Only)

GGUF is the format used by llama.cpp, ollama, and other efficient inference engines. It supports various quantization levels to reduce model size and memory usage.

### Basic Export

```ruby
llm = Fine::LLM.load("my_llm")
llm.export_gguf("model.gguf")
```

### Quantization Options

```ruby
# F16 - Good balance of quality and size (default)
llm.export_gguf("model-f16.gguf", quantization: :f16)

# Q8 - Smaller, minimal quality loss
llm.export_gguf("model-q8.gguf", quantization: :q8_0)

# Q4 - Smallest, some quality loss
llm.export_gguf("model-q4.gguf", quantization: :q4_0)
```

### Available Quantization Types

| Type | Description | Size Reduction | Quality |
|------|-------------|----------------|---------|
| `:f32` | 32-bit float | None | Lossless |
| `:f16` | 16-bit float | ~50% | Minimal loss |
| `:q8_0` | 8-bit quantization | ~75% | Very small loss |
| `:q4_0` | 4-bit quantization | ~87% | Noticeable loss |
| `:q4_k` | 4-bit K-quant | ~87% | Better than q4_0 |
| `:q5_k` | 5-bit K-quant | ~84% | Good balance |
| `:q6_k` | 6-bit K-quant | ~81% | High quality |

### Adding Metadata

```ruby
llm.export_gguf(
  "model.gguf",
  quantization: :q4_k,
  metadata: {
    "description" => "Fine-tuned on customer support data",
    "version" => "1.0.0",
    "author" => "Your Name"
  }
)
```

### Using with llama.cpp

```bash
# Run inference
./main -m model.gguf -p "What is Ruby?" -n 100

# Start a server
./server -m model.gguf --host 0.0.0.0 --port 8080
```

### Using with Ollama

```bash
# Create a Modelfile
echo 'FROM ./model.gguf' > Modelfile
echo 'PARAMETER temperature 0.7' >> Modelfile
echo 'SYSTEM "You are a helpful assistant."' >> Modelfile

# Create the model
ollama create my-model -f Modelfile

# Run it
ollama run my-model "What is Ruby?"
```

## Best Practices

### Choosing the Right Format

| Use Case | Recommended Format |
|----------|-------------------|
| Web deployment | ONNX |
| Mobile apps | ONNX (with quantization) |
| Server inference | ONNX or GGUF |
| Edge devices | GGUF (quantized) |
| llama.cpp / ollama | GGUF |

### Choosing Quantization

| Priority | Recommended |
|----------|------------|
| Quality first | `:f16` or `:q8_0` |
| Balance | `:q5_k` or `:q6_k` |
| Size first | `:q4_0` or `:q4_k` |

### Testing Exported Models

Always test your exported models to ensure quality:

```ruby
# Before export - test with Fine
response = llm.generate("Test prompt")

# After export - test with target runtime
# Compare outputs to ensure quality is acceptable
```

## Troubleshooting

### ONNX Export Fails

1. Ensure the model is loaded and trained
2. Check that torch.rb ONNX support is available
3. Try a different opset_version (11, 13, or 14)

### GGUF Export Issues

1. Only LLMs support GGUF export
2. Large models may need more memory during export
3. Some model architectures may need custom tensor mappings

### Large File Sizes

Use quantization to reduce file size:

```ruby
# ONNX with INT8 quantization
classifier.export_onnx("model.onnx", quantize: :int8)

# GGUF with Q4 quantization
llm.export_gguf("model.gguf", quantization: :q4_0)
```
