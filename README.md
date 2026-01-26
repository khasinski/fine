# Fine

Fine-tune machine learning models with Ruby.

```ruby
classifier = Fine::TextClassifier.new("distilbert-base-uncased")
classifier.fit(train_file: "reviews.jsonl", epochs: 3)
classifier.predict("This product is amazing!")
# => [{ label: "positive", score: 0.97 }]
```

## I want to fine-tune...

### Text classification (sentiment, spam, intent)

Classify text into categories—reviews, support tickets, chat messages.

```ruby
classifier = Fine::TextClassifier.new("distilbert-base-uncased")
classifier.fit(train_file: "data/reviews.jsonl", epochs: 3)

classifier.predict("Terrible experience, waste of money")
# => [{ label: "negative", score: 0.94 }]
```

[Full tutorial: Text Classification](docs/tutorials/text-classification.md)

---

### Text embeddings for semantic search

Train embeddings for your domain—legal docs, support tickets, product catalog.

```ruby
embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2")
embedder.fit(train_file: "data/pairs.jsonl", epochs: 3)

embedding = embedder.encode("How do I cancel my subscription?")
```

[Full tutorial: Text Embeddings](docs/tutorials/text-embeddings.md)

---

### Image classification

Classify images into categories (cats vs dogs, products, documents).

```ruby
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224")
classifier.fit(train_dir: "data/train", val_dir: "data/val", epochs: 3)

classifier.predict("photo.jpg")
# => [{ label: "cat", score: 0.95 }]
```

[Full tutorial: Image Classification](docs/tutorials/siglip2-image-classification.md)

---

### Image recognition for custom objects

Teach the model to recognize your products, logos, or custom objects.

```ruby
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-384") do |config|
  config.epochs = 5
  config.learning_rate = 1e-4
end

classifier.fit(train_dir: "products/train")
classifier.save("models/product_detector")
```

[Full tutorial: Object Recognition](docs/tutorials/siglip2-object-recognition.md)

---

### Image similarity search

Find visually similar images in your catalog.

```ruby
encoder = model.encoder
embedding = encoder.call(image_tensor)
similarity = cosine_similarity(embedding1, embedding2)
```

[Full tutorial: Similarity Search](docs/tutorials/siglip2-similarity-search.md)

---

### LLMs

Fine-tune Gemma, Llama, Qwen and other open models for custom tasks.

```ruby
llm = Fine::LLM.new("meta-llama/Llama-3.2-1B")
llm.fit(train_file: "instructions.jsonl", epochs: 3)

llm.generate("Explain Ruby blocks")
# => "A Ruby block is a chunk of code that can be passed to a method..."
```

[Full tutorial: LLM Fine-tuning](docs/tutorials/llm-fine-tuning.md)

---

### Tool Calling with LoRA

Train models to output Ollama-compatible tool calls using parameter-efficient LoRA.

```ruby
model = Fine::Models::CausalLM.from_pretrained("google/gemma-3-1b-it")

# Apply LoRA - only 0.5% of params trainable
Fine::LoRA.apply(model, rank: 32, alpha: 64, target_modules: %w[q_proj k_proj v_proj o_proj])
#   LoRA applied to 104 layers
#   Total params: 1.31B
#   Trainable params: 5.96M (0.46%)

# Train on tool calling data
lora_params = Fine::LoRA.trainable_parameters(model)
# ... training loop

# Output: valid Ollama JSON
# {"role":"assistant","tool_calls":[{"type":"function","function":{"index":0,"name":"get_weather","arguments":{"location":"Tokyo"}}}]}
```

[Full tutorial: LoRA Tool Calling](docs/tutorials/lora-tool-calling.md)

---

## Installation

```ruby
gem 'fine'
```

Requires Ruby 3.1+, LibTorch, and libvips.

[Full installation guide](docs/installation.md) | [Quickstart](docs/quickstart.md)

**Quick setup (macOS):**
```bash
brew install pytorch libvips
bundle install
```

## Supported Models

**Text Classification**

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| `distilbert-base-uncased` | 66M | Fast | Good |
| `bert-base-uncased` | 110M | Medium | Better |
| `microsoft/deberta-v3-small` | 44M | Fast | Great |
| `microsoft/deberta-v3-base` | 86M | Medium | Best |

**Text Embeddings**

| Model | Parameters | Best For |
|-------|------------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Fast, general purpose |
| `sentence-transformers/all-mpnet-base-v2` | 110M | Better quality |
| `BAAI/bge-small-en-v1.5` | 33M | Retrieval optimized |
| `BAAI/bge-base-en-v1.5` | 110M | Best retrieval |

**Vision (SigLIP2)**

| Model | Parameters | Best For |
|-------|------------|----------|
| `google/siglip2-base-patch16-224` | 86M | Quick experiments |
| `google/siglip2-base-patch16-384` | 86M | Good balance |
| `google/siglip2-large-patch16-256` | 303M | Maximum accuracy |
| `google/siglip2-so400m-patch14-224` | 400M | Best quality |

**LLMs**

| Model | Parameters | Best For |
|-------|------------|----------|
| `google/gemma-3-1b-it` | 1B | Fast experiments, tool calling |
| `meta-llama/Llama-3.2-1B` | 1B | Fast experiments |
| `google/gemma-3-4b-it` | 4B | Good balance |
| `Qwen/Qwen2-1.5B` | 1.5B | Multilingual |
| `mistralai/Mistral-7B-v0.1` | 7B | Best quality |

## Configuration

```ruby
Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  config.epochs = 3
  config.batch_size = 16
  config.learning_rate = 2e-5
  config.max_length = 256

  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch}: #{metrics[:accuracy]}"
  end
end
```

## Export for Deployment

Export fine-tuned models to production formats.

**ONNX** - For ONNX Runtime, TensorRT, OpenVINO:

```ruby
classifier.export_onnx("model.onnx")
embedder.export_onnx("embedder.onnx")
```

**GGUF** - For llama.cpp, ollama (LLMs only):

```ruby
llm.export_gguf("model.gguf", quantization: :q4_0)
```

[Full tutorial: Model Export](docs/tutorials/model-export.md)

## Roadmap

- [x] SigLIP2 image classification
- [x] Text classification (BERT, DeBERTa)
- [x] Text embedding models
- [x] LLM fine-tuning (Gemma, Llama, Qwen)
- [x] ONNX & GGUF export
- [x] LoRA fine-tuning
- [ ] QLoRA (4-bit quantized LoRA)

## Contributing

Bug reports and pull requests welcome at [github.com/khasinski/fine](https://github.com/khasinski/fine).

## License

MIT
