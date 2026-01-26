#!/usr/bin/env ruby
# frozen_string_literal: true

# Train larger model with LoRA for efficient fine-tuning
# LoRA dramatically reduces memory by only training ~1% of parameters

require "bundler/setup"
require "fine"

MAX_MEMORY_GB = 40
MONITOR_INTERVAL = 2

def get_memory_usage_gb
  `ps -o rss= -p #{Process.pid}`.strip.to_i / 1024.0 / 1024.0
end

puts "=" * 70
puts "LORA TOOL CALLING TRAINING"
puts "=" * 70
puts "Max memory limit: #{MAX_MEMORY_GB} GB"

max_memory_seen = 0.0
memory_exceeded = false

monitor_thread = Thread.new do
  loop do
    mem = get_memory_usage_gb
    max_memory_seen = mem if mem > max_memory_seen
    if mem > MAX_MEMORY_GB
      memory_exceeded = true
      Thread.main.raise(Interrupt, "Memory limit exceeded: #{mem.round(2)} GB")
    end
    sleep MONITOR_INTERVAL
  rescue => e
    break if e.is_a?(Interrupt)
  end
end

begin
  Fine.configure { |c| c.progress_bar = false }

  # Use larger dataset
  data_path = File.expand_path("data/ollama_tool_calls_large.jsonl", __dir__)

  # Try 4B model first, fall back to 1B if memory issues
  model_id = ARGV[0] || "google/gemma-3-1b-it"

  puts "\n[1/6] Loading model: #{model_id}..."
  model = Fine::Models::CausalLM.from_pretrained(model_id)
  puts "   Model loaded: #{get_memory_usage_gb.round(2)} GB"

  puts "\n[2/6] Applying LoRA..."
  # Apply LoRA to attention projections
  Fine::LoRA.apply(
    model,
    rank: 32,           # Higher rank = more capacity for structured output
    alpha: 64,          # Scaling factor
    dropout: 0.05,      # Light dropout for regularization
    target_modules: %w[q_proj k_proj v_proj o_proj]  # All attention projections
  )
  puts "   LoRA applied: #{get_memory_usage_gb.round(2)} GB"

  # Move to device
  model.to(Fine.device)
  model.train
  puts "   On #{Fine.device}: #{get_memory_usage_gb.round(2)} GB"

  puts "\n[3/6] Loading tokenizer..."
  downloader = Fine::Hub::ModelDownloader.new(model_id)
  model_path = downloader.download
  tokenizer = Fine::Tokenizers::AutoTokenizer.new(model_path, max_length: 384)
  puts "   Found tokenizer"

  puts "\n[4/6] Loading training data..."
  dataset = Fine::Datasets::InstructionDataset.from_jsonl(
    data_path,
    tokenizer: tokenizer,
    format: :alpaca,
    max_length: 384
  )
  puts "   #{dataset.size} examples loaded"

  data_loader = Fine::Datasets::InstructionDataLoader.new(
    dataset,
    batch_size: 1,
    shuffle: true,
    pad_token_id: tokenizer.pad_token_id
  )

  puts "\n[5/6] Training with LoRA..."

  # Only get LoRA parameters for optimizer
  lora_params = Fine::LoRA.trainable_parameters(model)
  optimizer = Torch::Optim::AdamW.new(lora_params, lr: 1e-4)  # Higher LR for LoRA

  epochs = 15  # More epochs for structured output learning
  total_loss = 0.0
  step = 0

  epochs.times do |epoch|
    epoch_loss = 0.0
    batch_count = 0

    data_loader.each do |batch|
      input_ids = batch[:input_ids].to(Fine.device)
      labels = batch[:labels].to(Fine.device)
      attention_mask = batch[:attention_mask].to(Fine.device)

      # Forward
      outputs = model.forward(input_ids, attention_mask: attention_mask, labels: labels)
      loss = outputs[:loss]

      # Backward
      loss.backward

      # Optimizer step
      optimizer.step
      optimizer.zero_grad

      epoch_loss += loss.to(:float32).item
      batch_count += 1
      step += 1
    end

    avg_loss = epoch_loss / batch_count
    mem = get_memory_usage_gb
    puts "   Epoch #{epoch + 1}: loss=#{avg_loss.round(4)} | Memory: #{mem.round(2)} GB"
  end

  puts "\n[6/6] Testing generation..."

  model.eval
  test_cases = [
    {
      prompt: "What's the weather in Tokyo?",
      tools: "get_weather: Get current weather\n  Parameters: location (string, required)"
    },
    {
      prompt: "Calculate 50 + 25 * 2",
      tools: "calculate: Math calculator\n  Parameters: expression (string, required)"
    },
    {
      prompt: "Search for Ruby tutorials",
      tools: "search_web: Web search\n  Parameters: query (string, required)"
    }
  ]

  test_cases.each do |tc|
    full_prompt = <<~PROMPT
### Instruction:
#{tc[:prompt]}

### Input:
You have access to the following tools:

#{tc[:tools]}

Respond with a JSON tool call if a tool is needed.

### Response:
PROMPT

    ids = tokenizer.encode_for_generation(full_prompt)
    input_ids = Torch.tensor([ids]).to(Fine.device)

    Torch.no_grad do
      output_ids = model.generate(
        input_ids,
        max_new_tokens: 150,
        temperature: 0.1,
        do_sample: false,
        eos_token_id: tokenizer.eos_token_id
      )
      response = tokenizer.decode(output_ids[0].to_a)
      generated = response.split("### Response:").last.to_s.strip

      puts "\n   Q: #{tc[:prompt]}"
      puts "   A: #{generated[0..200]}"

      begin
        json = JSON.parse(generated)
        if json["tool_calls"]
          puts "   [Valid Ollama format]"
        end
      rescue JSON::ParserError
        puts "   [Not valid JSON]"
      end
    end
  end

  puts "\n" + "=" * 70
  save_path = "/tmp/gemma3-lora-tools"

  # Merge LoRA weights for inference
  puts "Merging LoRA weights..."
  Fine::LoRA.merge!(model)

  model.save(save_path)
  tokenizer.save(save_path)
  puts "Model saved to: #{save_path}"
  puts "Max memory used: #{max_memory_seen.round(2)} GB"
  puts "=" * 70

rescue Interrupt => e
  if memory_exceeded
    puts "\n\nTERMINATED: Memory limit exceeded!"
    exit 1
  else
    puts "\n\nInterrupted"
    exit 130
  end
rescue => e
  puts "\nFailed: #{e.class}: #{e.message}"
  puts e.backtrace.first(10).join("\n")
  exit 1
ensure
  monitor_thread&.kill
end
