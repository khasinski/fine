# Quickstart

Get started with Fine in under 5 minutes.

## Installation

```bash
# macOS
brew install pytorch libvips
gem install fine

# Or add to Gemfile
gem 'fine'
```

## Text Classification

Classify text into categories (sentiment, spam, intent).

**1. Prepare your data** (`reviews.jsonl`):
```json
{"text": "This product is amazing!", "label": "positive"}
{"text": "Terrible experience, waste of money", "label": "negative"}
{"text": "It's okay, nothing special", "label": "neutral"}
```

**2. Train and use:**
```ruby
require 'fine'

classifier = Fine::TextClassifier.new("distilbert-base-uncased")
classifier.fit(train_file: "reviews.jsonl", epochs: 3)

classifier.predict("Best purchase ever!")
# => [{ label: "positive", score: 0.95 }]

classifier.save("my_classifier")
```

---

## Image Classification

Classify images into categories.

**1. Organize your images:**
```
data/
  cats/
    cat1.jpg
    cat2.jpg
  dogs/
    dog1.jpg
    dog2.jpg
```

**2. Train and use:**
```ruby
require 'fine'

classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224")
classifier.fit(train_dir: "data/", epochs: 3)

classifier.predict("test_image.jpg")
# => [{ label: "cat", score: 0.92 }]

classifier.save("my_image_classifier")
```

---

## Text Embeddings

Generate embeddings for semantic search.

**1. Prepare training pairs** (`pairs.jsonl`):
```json
{"query": "How do I reset my password?", "positive": "Click 'Forgot Password' on the login page"}
{"query": "What are your hours?", "positive": "We're open Monday-Friday, 9am-5pm"}
```

**2. Train and use:**
```ruby
require 'fine'

embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2")
embedder.fit(train_file: "pairs.jsonl", epochs: 3)

# Get embeddings
embedding = embedder.encode("How do I change my password?")

# Semantic search
results = embedder.search("password help", corpus, top_k: 5)

embedder.save("my_embedder")
```

---

## LLM Fine-tuning (with LoRA)

Fine-tune language models using LoRA—only 0.5% of parameters are trainable.

**1. Prepare instruction data** (`instructions.jsonl`):
```json
{"instruction": "Summarize this text", "input": "Long article here...", "output": "Brief summary"}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

**2. Train with LoRA:**
```ruby
require 'fine'

# Load model and apply LoRA
model = Fine::Models::CausalLM.from_pretrained("google/gemma-3-1b-it")
Fine::LoRA.apply(model, rank: 32, alpha: 64)
# => "Trainable params: 5.96M (0.46%)"

# Load dataset
tokenizer = Fine::Tokenizers::AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
dataset = Fine::Datasets::InstructionDataset.from_jsonl("instructions.jsonl", tokenizer: tokenizer)

# Train
config = Fine::LLMConfiguration.new
trainer = Fine::Training::LLMTrainer.new(model, config, train_dataset: dataset)
trainer.fit

# Merge LoRA weights and save
Fine::LoRA.merge!(model)
model.save("my_llm")
```

---

## Data Formats Reference

### Text Classification
```json
{"text": "Your text here", "label": "category_name"}
```

### Text Pairs (Embeddings)
```json
{"query": "Question text", "positive": "Matching answer"}
```
Alternative field names: `text_a`/`text_b`, `anchor`/`positive`, `sentence1`/`sentence2`

### Instructions (LLM)

**Alpaca format:**
```json
{"instruction": "Task description", "input": "Optional context", "output": "Expected response"}
```

**ShareGPT format:**
```json
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "assistant", "value": "Hello!"}]}
```

**Simple format:**
```json
{"prompt": "Input text", "completion": "Output text"}
```

---

## Configuration

All classifiers accept a configuration block:

```ruby
Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  config.epochs = 5
  config.batch_size = 16
  config.learning_rate = 2e-5

  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch}: loss=#{metrics[:loss]}"
  end
end
```

Common options:
- `epochs` - Number of training passes (default: 3)
- `batch_size` - Samples per batch (default: 8-16)
- `learning_rate` - Learning rate (default: 2e-5)
- `max_length` - Max sequence length (default: 128-2048)

---

## Export for Production

```ruby
# ONNX (for ONNX Runtime, TensorRT)
classifier.export_onnx("model.onnx")

# GGUF (for llama.cpp, Ollama)
llm.export_gguf("model.gguf", quantization: :q4_0)
```

---

## Next Steps

- [Text Classification Tutorial](tutorials/text-classification.md)
- [Image Classification Tutorial](tutorials/siglip2-image-classification.md)
- [LLM Fine-tuning Tutorial](tutorials/llm-fine-tuning.md)
- [LoRA Tool Calling](tutorials/lora-tool-calling.md)
- [Model Export](tutorials/model-export.md)
