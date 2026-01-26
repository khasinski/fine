#!/usr/bin/env ruby
# frozen_string_literal: true

# Fine-tune Gemma 3 (1B) for tool calling
#
# This script fine-tunes Gemma 3 1B instruction-tuned model to generate
# tool calls in Ollama-compatible format.
#
# Format:
#   Input:  [AVAILABLE_TOOLS] [{...tool definitions...}] [/AVAILABLE_TOOLS]
#   Output: [TOOL_CALLS] [{...tool calls...}]

require "bundler/setup"
require "fine"

puts "=" * 60
puts "GEMMA 3 TOOL CALLING FINE-TUNING"
puts "=" * 60

Fine.configure do |config|
  config.progress_bar = false
end

data_path = File.expand_path("data/tool_calls.jsonl", __dir__)

# Gemma 3 1B instruction-tuned
model_id = "google/gemma-3-1b-it"

puts "\n1. Setting up LLM with #{model_id}..."
puts "   (This will download ~2GB from HuggingFace if not cached)"

begin
  llm = Fine::LLM.new(model_id) do |config|
    config.epochs = 3
    config.batch_size = 1  # Reduced for memory
    config.learning_rate = 2e-5
    config.max_length = 256  # Reduced for memory
    config.gradient_accumulation_steps = 1  # Set to 1 to avoid memory issues
    config.warmup_steps = 10
    config.max_grad_norm = nil  # Disable gradient clipping to simplify

    config.on_epoch_end do |epoch, metrics|
      puts "    Epoch #{epoch}: loss=#{metrics[:loss].round(4)}"
    end
  end

  puts "   Config:"
  puts "     Epochs: #{llm.config.epochs}"
  puts "     Batch size: #{llm.config.batch_size}"
  puts "     Learning rate: #{llm.config.learning_rate}"
  puts "     Max length: #{llm.config.max_length}"

  puts "\n2. Loading training data from #{data_path}..."
  line_count = File.readlines(data_path).count
  puts "   Found #{line_count} examples"

  puts "\n3. Starting fine-tuning..."
  puts "   (This may take a while depending on your hardware)"
  puts ""

  history = llm.fit(train_file: data_path, format: :alpaca)

  puts "\n4. Training completed!"
  puts "   Final loss: #{history.last[:loss].round(4)}"

  if history.size >= 2 && history.last[:loss] < history.first[:loss]
    improvement = ((1 - history.last[:loss] / history.first[:loss]) * 100).round(1)
    puts "   Loss improved by #{improvement}%"
  end

  puts "\n5. Testing tool call generation..."

  test_prompts = [
    {
      instruction: "What's the weather in Chicago?",
      tools: "Available tools:\n- get_weather(location: string) - Get current weather\n- calculate(expression: string) - Evaluate math"
    },
    {
      instruction: "Calculate 50 + 25 * 2",
      tools: "Available tools:\n- get_weather(location: string) - Get current weather\n- calculate(expression: string) - Evaluate math"
    },
    {
      instruction: "Search for Python tutorials",
      tools: "Available tools:\n- search_web(query: string) - Search the web\n- get_weather(location: string) - Get weather"
    }
  ]

  test_prompts.each do |prompt|
    full_prompt = "### Instruction:\n#{prompt[:instruction]}\n\n### Input:\n#{prompt[:tools]}\n\n### Response:\n"

    response = llm.generate(
      full_prompt,
      max_new_tokens: 100,
      temperature: 0.1,  # Low temperature for deterministic tool calls
      do_sample: false   # Greedy decoding for consistent output
    )

    # Extract just the response part
    generated = response.split("### Response:").last.strip

    puts "\n   Prompt: \"#{prompt[:instruction]}\""
    puts "   Generated:"
    puts "   #{generated.lines.first(5).map(&:strip).join("\n   ")}"
  end

  puts "\n6. Saving fine-tuned model..."
  save_path = "/tmp/gemma3-tool-calling"
  llm.save(save_path)
  puts "   Saved to: #{save_path}"

  puts "\n7. Testing load and generate..."
  loaded = Fine::LLM.load(save_path)

  test_prompt = "### Instruction:\nCheck the temperature in Boston\n\n### Input:\nAvailable tools:\n- get_weather(location: string) - Get weather\n\n### Response:\n"
  loaded_response = loaded.generate(test_prompt, max_new_tokens: 50, do_sample: false)
  generated = loaded_response.split("### Response:").last.strip

  puts "   Loaded model response:"
  puts "   #{generated.lines.first(3).map(&:strip).join("\n   ")}"

  puts "\n" + "=" * 60
  puts "GEMMA 3 TOOL CALLING FINE-TUNING COMPLETE!"
  puts "=" * 60
  puts "\nModel saved to: #{save_path}"
  puts "You can load it with: Fine::LLM.load('#{save_path}')"

rescue => e
  puts "\n" + "=" * 60
  puts "FINE-TUNING FAILED!"
  puts "=" * 60
  puts "\nError: #{e.class}: #{e.message}"
  puts "\nBacktrace:"
  puts e.backtrace.first(20).join("\n")
  exit 1
end
