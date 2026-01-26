#!/usr/bin/env ruby
# frozen_string_literal: true

# Test LLM components (loading, forward pass, minimal training)
# Note: Full LLM training requires significant compute

require "bundler/setup"
require "fine"

puts "=" * 60
puts "LLM COMPONENTS TEST"
puts "=" * 60

Fine.configure do |config|
  config.progress_bar = false
end

fixtures_path = File.expand_path("../spec/fixtures/text/instructions.jsonl", __dir__)

puts "\n1. Testing LlamaDecoder components..."

begin
  # Test with a small config to verify components work
  test_config = Fine::Hub::ConfigLoader.new("/dev/null") rescue nil

  # Create a minimal config for testing
  class MinimalConfig
    attr_accessor :config

    def initialize
      @config = {}
    end

    def vocab_size = 1000
    def hidden_size = 256
    def num_hidden_layers = 2
    def num_attention_heads = 4
    def intermediate_size = 512
    def max_position_embeddings = 128
    def rms_norm_eps = 1e-6
    def rope_theta = 10000.0
    def num_key_value_heads = 4
    def use_bias = false
    def to_h = @config
  end

  config = MinimalConfig.new

  puts "   Creating LlamaDecoder with small config..."
  decoder = Fine::Models::LlamaDecoder.new(config)
  puts "   ✓ LlamaDecoder created"

  puts "   Testing forward pass..."
  input_ids = Torch.randint(0, 1000, [2, 16], dtype: :long)  # batch=2, seq=16
  output = decoder.call(input_ids)
  puts "   ✓ Forward pass successful"
  puts "     Input shape: #{input_ids.shape.to_a}"
  puts "     Output shape: #{output[:last_hidden_state].shape.to_a}"

  puts "\n2. Testing CausalLM wrapper..."
  lm = Fine::Models::CausalLM.new(config)
  output = lm.call(input_ids)
  puts "   ✓ CausalLM forward pass successful"
  puts "     Logits shape: #{output[:logits].shape.to_a}"

  puts "\n3. Testing with labels (loss computation)..."
  labels = input_ids.clone
  output = lm.call(input_ids, labels: labels)
  puts "   ✓ Loss computed: #{output[:loss].item.round(4)}"

  puts "\n4. Testing InstructionDataset..."
  tokenizer_mock = Object.new
  def tokenizer_mock.encode(text, **_)
    ids = Array.new(10) { rand(100) }
    { input_ids: [ids], attention_mask: [Array.new(10, 1)] }
  end
  def tokenizer_mock.pad_token_id = 0
  def tokenizer_mock.eos_token_id = 2

  dataset = Fine::Datasets::InstructionDataset.from_jsonl(
    fixtures_path,
    tokenizer: tokenizer_mock,
    max_length: 32
  )
  puts "   ✓ Dataset loaded with #{dataset.size} examples"

  item = dataset[0]
  puts "   Sample item keys: #{item.keys}"
  puts "   Input IDs length: #{item[:input_ids].size}"

  puts "\n5. Testing training step..."
  optimizer = Torch::Optim::Adam.new(lm.parameters, lr: 1e-4)

  initial_loss = nil
  3.times do |i|
    optimizer.zero_grad

    batch_ids = Torch.randint(0, 1000, [2, 16], dtype: :long)
    labels = batch_ids.clone
    output = lm.call(batch_ids, labels: labels)

    loss = output[:loss]
    initial_loss ||= loss.item
    loss.backward
    optimizer.step

    puts "   Step #{i + 1}: loss=#{loss.item.round(4)}"
  end

  if output[:loss].item < initial_loss
    puts "   ✓ Loss decreased during training"
  else
    puts "   ⚠ Loss did not decrease (expected with random data)"
  end

  puts "\n" + "=" * 60
  puts "LLM COMPONENTS TEST PASSED!"
  puts "=" * 60

rescue => e
  puts "\n" + "=" * 60
  puts "LLM COMPONENTS TEST FAILED!"
  puts "=" * 60
  puts "\nError: #{e.class}: #{e.message}"
  puts "\nBacktrace:"
  puts e.backtrace.first(15).join("\n")
  exit 1
end
