# LoRA Fine-Tuning for Tool Calling

Train LLMs to generate Ollama-compatible tool calls using parameter-efficient LoRA fine-tuning.

## Overview

This tutorial shows how to fine-tune Gemma 3 to output structured JSON tool calls that work with Ollama's tool calling API. Using LoRA, we train only 0.5% of the model's parameters while achieving good results.

**What you'll learn:**
- Prepare training data in Ollama tool call format
- Apply LoRA for memory-efficient fine-tuning
- Train and evaluate the model
- Export for use with Ollama

## Quick Start

```ruby
require "fine"

# Load model
model = Fine::Models::CausalLM.from_pretrained("google/gemma-3-1b-it")

# Apply LoRA (only 0.5% of params become trainable)
Fine::LoRA.apply(model, rank: 32, alpha: 64, target_modules: %w[q_proj k_proj v_proj o_proj])

# Train
llm = Fine::LLM.new("google/gemma-3-1b-it")
llm.fit(train_file: "tool_calls.jsonl", epochs: 15)

# Generate
llm.generate("What's the weather in Tokyo?")
# => {"role":"assistant","tool_calls":[{"type":"function","function":{"index":0,"name":"get_weather","arguments":{"location":"Tokyo"}}}]}
```

## Training Data Format

Ollama expects tool calls in this JSON format:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "index": 0,
        "name": "get_weather",
        "arguments": {
          "location": "Tokyo"
        }
      }
    }
  ]
}
```

Create training data in Alpaca format with this output structure:

```jsonl
{"instruction": "What's the weather in Tokyo?", "input": "You have access to the following tools:\n\nget_weather: Get current weather\n  Parameters: location (string, required)\n\nRespond with a JSON tool call if a tool is needed.", "output": "{\"role\": \"assistant\", \"tool_calls\": [{\"type\": \"function\", \"function\": {\"index\": 0, \"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}}]}"}
{"instruction": "Calculate 25 * 4", "input": "You have access to the following tools:\n\ncalculate: Math calculator\n  Parameters: expression (string, required)\n\nRespond with a JSON tool call if a tool is needed.", "output": "{\"role\": \"assistant\", \"tool_calls\": [{\"type\": \"function\", \"function\": {\"index\": 0, \"name\": \"calculate\", \"arguments\": {\"expression\": \"25 * 4\"}}}]}"}
```

## Full Training Script

```ruby
#!/usr/bin/env ruby
require "bundler/setup"
require "fine"

Fine.configure { |c| c.progress_bar = false }

model_id = "google/gemma-3-1b-it"
data_path = "data/ollama_tool_calls.jsonl"

# 1. Load model
puts "Loading model..."
model = Fine::Models::CausalLM.from_pretrained(model_id)

# 2. Apply LoRA
puts "Applying LoRA..."
Fine::LoRA.apply(
  model,
  rank: 32,           # Higher rank = more capacity
  alpha: 64,          # Scaling factor (usually 2x rank)
  dropout: 0.05,      # Light regularization
  target_modules: %w[q_proj k_proj v_proj o_proj]
)
# Output:
#   LoRA applied to 104 layers
#   Total params: 1.31B
#   Trainable params: 5.96M (0.46%)

# 3. Move to GPU and set train mode
model.to(Fine.device)
model.train

# 4. Load tokenizer and data
downloader = Fine::Hub::ModelDownloader.new(model_id)
model_path = downloader.download
tokenizer = Fine::Tokenizers::AutoTokenizer.new(model_path, max_length: 384)

dataset = Fine::Datasets::InstructionDataset.from_jsonl(
  data_path,
  tokenizer: tokenizer,
  format: :alpaca,
  max_length: 384
)

data_loader = Fine::Datasets::InstructionDataLoader.new(
  dataset,
  batch_size: 1,
  shuffle: true,
  pad_token_id: tokenizer.pad_token_id
)

# 5. Train with LoRA parameters only
lora_params = Fine::LoRA.trainable_parameters(model)
optimizer = Torch::Optim::AdamW.new(lora_params, lr: 1e-4)

15.times do |epoch|
  epoch_loss = 0.0
  batch_count = 0

  data_loader.each do |batch|
    input_ids = batch[:input_ids].to(Fine.device)
    labels = batch[:labels].to(Fine.device)
    attention_mask = batch[:attention_mask].to(Fine.device)

    outputs = model.forward(input_ids, attention_mask: attention_mask, labels: labels)
    loss = outputs[:loss]

    loss.backward
    optimizer.step
    optimizer.zero_grad

    epoch_loss += loss.to(:float32).item
    batch_count += 1
  end

  puts "Epoch #{epoch + 1}: loss=#{(epoch_loss / batch_count).round(4)}"
end

# 6. Merge LoRA weights and save
Fine::LoRA.merge!(model)
model.save("/tmp/gemma3-tools")
tokenizer.save("/tmp/gemma3-tools")
```

## Training Results

With 105 training examples and 15 epochs:

| Metric | Value |
|--------|-------|
| Initial loss | 3.78 |
| Final loss | 0.106 |
| Improvement | 97% |
| Training memory | 4.8-5.0 GB |
| Peak memory | ~10 GB |
| Trainable params | 5.96M (0.46%) |

## LoRA Configuration

### Rank

Controls model capacity. Higher = more expressive but more memory.

| Rank | Trainable Params | Use Case |
|------|------------------|----------|
| 8 | ~1.5M | Simple tasks |
| 16 | ~3M | General fine-tuning |
| 32 | ~6M | Complex structured output |
| 64 | ~12M | Maximum capacity |

### Target Modules

Which layers to apply LoRA to:

```ruby
# Minimal (query + value only)
target_modules: %w[q_proj v_proj]

# Recommended (all attention)
target_modules: %w[q_proj k_proj v_proj o_proj]

# Maximum (attention + MLP)
target_modules: %w[q_proj k_proj v_proj o_proj gate_proj up_proj down_proj]
```

### Alpha

Scaling factor, typically 2x the rank:

```ruby
Fine::LoRA.apply(model, rank: 32, alpha: 64)  # alpha = 2 * rank
```

## Memory Usage

LoRA dramatically reduces memory compared to full fine-tuning:

| Model | Full Fine-tune | LoRA (rank=32) |
|-------|----------------|----------------|
| Gemma 3 1B | ~12 GB | ~5 GB |
| Gemma 3 4B | ~40 GB | ~12 GB |
| Llama 3.2 7B | ~60 GB | ~18 GB |

## Testing Generation

```ruby
model.eval

prompt = <<~PROMPT
### Instruction:
What's the weather in Tokyo?

### Input:
You have access to the following tools:

get_weather: Get current weather
  Parameters: location (string, required)

Respond with a JSON tool call if a tool is needed.

### Response:
PROMPT

ids = tokenizer.encode_for_generation(prompt)
input_ids = Torch.tensor([ids]).to(Fine.device)

Torch.no_grad do
  output_ids = model.generate(
    input_ids,
    max_new_tokens: 150,
    temperature: 0.1,
    do_sample: false
  )

  response = tokenizer.decode(output_ids[0].to_a)
  json_output = response.split("### Response:").last.strip

  puts json_output
  # {"role":"assistant","tool_calls":[{"type":"function","function":{"index":0,"name":"get_weather","arguments":{"location":"Tokyo"}}}]}
end
```

## Tips for Better Results

1. **More training data** - 100+ examples per tool type
2. **Diverse examples** - Vary phrasing and argument values
3. **Higher rank** - Use rank 32-64 for complex JSON
4. **More epochs** - 15-20 epochs for structured output
5. **Lower learning rate** - 1e-4 to 5e-5 for stability

## Generating Training Data

Use this script to generate diverse examples:

```ruby
TOOLS = {
  get_weather: {
    description: "Get current weather",
    params: { location: "string, required" },
    examples: [
      { q: "What's the weather in Tokyo?", args: { location: "Tokyo" } },
      { q: "Is it raining in Seattle?", args: { location: "Seattle" } },
      # ... more examples
    ]
  },
  calculate: {
    description: "Math calculator",
    params: { expression: "string, required" },
    examples: [
      { q: "Calculate 25 * 4", args: { expression: "25 * 4" } },
      # ... more examples
    ]
  }
}

def generate_output(name, args)
  {
    role: "assistant",
    tool_calls: [{
      type: "function",
      function: { index: 0, name: name.to_s, arguments: args }
    }]
  }
end

# Generate JSONL
TOOLS.each do |name, tool|
  tool[:examples].each do |ex|
    puts({
      instruction: ex[:q],
      input: "You have access to...",
      output: generate_output(name, ex[:args]).to_json
    }.to_json)
  end
end
```

## See Also

- [LLM Fine-tuning](llm-fine-tuning.md) - Full fine-tuning without LoRA
- [Model Export](model-export.md) - Export to GGUF for Ollama
