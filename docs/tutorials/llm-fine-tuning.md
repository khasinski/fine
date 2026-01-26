# Fine-tuning LLMs with Fine

Fine supports fine-tuning open-source LLMs like Llama, Gemma, Mistral, and Qwen for custom tasks.

## Quick Start

```ruby
require "fine"

# Load a base model
llm = Fine::LLM.new("meta-llama/Llama-3.2-1B")

# Fine-tune on your data
llm.fit(train_file: "instructions.jsonl", epochs: 3)

# Generate text
response = llm.generate("Explain Ruby blocks in simple terms")
puts response

# Save for later
llm.save("my_llama")
```

## Preparing Your Data

Fine supports multiple data formats.

### Alpaca Format (Recommended)

```jsonl
{"instruction": "Explain what a Ruby block is", "input": "", "output": "A Ruby block is a chunk of code..."}
{"instruction": "Convert this to Ruby", "input": "print('hello')", "output": "puts 'hello'"}
{"instruction": "What does this code do?", "input": "arr.map(&:upcase)", "output": "It converts each string in the array to uppercase."}
```

### ShareGPT Format

```jsonl
{"conversations": [{"from": "human", "value": "What is Ruby?"}, {"from": "gpt", "value": "Ruby is a dynamic programming language..."}]}
{"conversations": [{"from": "human", "value": "Show me a loop"}, {"from": "gpt", "value": "Here's a Ruby loop:\n\n```ruby\n5.times { |i| puts i }\n```"}]}
```

### Simple Format

```jsonl
{"prompt": "### Question: What is 2+2?\n### Answer:", "completion": " 4"}
{"prompt": "Translate to French: Hello", "completion": " Bonjour"}
```

## Configuration Options

```ruby
llm = Fine::LLM.new("google/gemma-2b") do |config|
  # Training parameters
  config.epochs = 3
  config.batch_size = 4
  config.learning_rate = 2e-5

  # Sequence length
  config.max_length = 2048

  # Gradient accumulation (effective batch = batch_size * gradient_accumulation_steps)
  config.gradient_accumulation_steps = 4

  # Gradient clipping
  config.max_grad_norm = 1.0

  # Learning rate warmup
  config.warmup_steps = 100

  # Freeze bottom N layers (for faster training)
  config.freeze_layers = 8
end
```

## Supported Models

| Model Family | Example Model ID | Notes |
|-------------|------------------|-------|
| Llama 3.2 | `meta-llama/Llama-3.2-1B` | Great balance of size/quality |
| Gemma | `google/gemma-2b` | Good for instruction following |
| Mistral | `mistralai/Mistral-7B-v0.1` | Strong general performance |
| Qwen | `Qwen/Qwen2-1.5B` | Multilingual support |

## Training Strategies

### Full Fine-tuning

Train all parameters (requires more memory):

```ruby
llm = Fine::LLM.new("meta-llama/Llama-3.2-1B") do |config|
  config.freeze_layers = 0  # Train everything
  config.batch_size = 2     # Smaller batch for memory
  config.gradient_accumulation_steps = 8
end
```

### Partial Fine-tuning

Freeze early layers to reduce memory and training time:

```ruby
llm = Fine::LLM.new("meta-llama/Llama-3.2-1B") do |config|
  config.freeze_layers = 16  # Freeze bottom 16 layers
  config.batch_size = 4
end
```

## Generation Options

```ruby
# Load trained model
llm = Fine::LLM.load("my_llama")

# Greedy decoding (deterministic)
response = llm.generate(
  "What is Ruby?",
  do_sample: false
)

# Creative generation
response = llm.generate(
  "Write a poem about coding",
  temperature: 0.9,
  top_p: 0.95,
  max_new_tokens: 200
)

# Focused generation
response = llm.generate(
  "Explain recursion",
  temperature: 0.3,
  top_k: 10,
  max_new_tokens: 150
)
```

## Chat Interface

For conversational use:

```ruby
messages = [
  { role: "system", content: "You are a helpful Ruby programming assistant." },
  { role: "user", content: "How do I read a file in Ruby?" }
]

response = llm.chat(messages, max_new_tokens: 200)
puts response
```

## Memory Optimization

LLMs require significant memory. Here are strategies to reduce usage:

### 1. Use Gradient Accumulation

```ruby
config.batch_size = 1
config.gradient_accumulation_steps = 16
# Effective batch size = 16, but only 1 sample in memory at a time
```

### 2. Freeze Layers

```ruby
config.freeze_layers = 20  # Only train top layers
```

### 3. Reduce Sequence Length

```ruby
config.max_length = 512  # Instead of default 2048
```

### 4. Use Smaller Models

Start with 1B-3B parameter models:
- `meta-llama/Llama-3.2-1B`
- `google/gemma-2b`
- `Qwen/Qwen2-1.5B`

## Example: Code Assistant

```ruby
require "fine"

# Prepare data (code_instructions.jsonl)
# {"instruction": "Write a function to reverse a string", "input": "", "output": "def reverse(s)\n  s.reverse\nend"}

llm = Fine::LLM.new("meta-llama/Llama-3.2-1B") do |config|
  config.epochs = 3
  config.batch_size = 2
  config.max_length = 1024
  config.freeze_layers = 8
end

llm.fit(train_file: "code_instructions.jsonl")
llm.save("ruby_code_assistant")

# Use it
assistant = Fine::LLM.load("ruby_code_assistant")
code = assistant.generate(
  "### Instruction:\nWrite a function to find the maximum element in an array\n\n### Response:\n",
  temperature: 0.2,
  max_new_tokens: 200
)
puts code
```

## Example: Domain Expert

```ruby
# Fine-tune on domain-specific Q&A
llm = Fine::LLM.new("google/gemma-2b") do |config|
  config.epochs = 5
  config.learning_rate = 1e-5
end

llm.fit(train_file: "medical_qa.jsonl", format: :alpaca)
llm.save("medical_assistant")
```

## Evaluation

Track training with validation data:

```ruby
llm.fit(
  train_file: "train.jsonl",
  val_file: "val.jsonl",
  epochs: 3
)
# Logs val_loss and val_perplexity each epoch
```

Lower perplexity indicates better language modeling.

## Tips

1. **Start small**: Begin with 1B models and small datasets
2. **Quality over quantity**: 1000 high-quality examples often beats 10000 noisy ones
3. **Format consistency**: Keep your instruction format consistent
4. **Learning rate**: Use lower rates (1e-5 to 5e-5) for fine-tuning
5. **Early stopping**: Monitor validation loss to avoid overfitting
