#!/usr/bin/env ruby
# frozen_string_literal: true

# Quick test of LLM components with local fixtures

require "bundler/setup"
require "fine"

puts "Testing Fine::LLM components..."
puts "=" * 50

# Use test fixtures
fixtures_path = File.expand_path("../spec/fixtures/text/instructions.jsonl", __dir__)

puts "\n1. Creating LLM..."
llm = Fine::LLM.new("meta-llama/Llama-3.2-1B") do |config|
  config.epochs = 1
  config.batch_size = 1
  config.learning_rate = 1e-5
  config.max_length = 128
end

puts "   Config: epochs=#{llm.config.epochs}, batch_size=#{llm.config.batch_size}, max_length=#{llm.config.max_length}"

puts "\n2. Testing InstructionDataset loading..."

# Create a mock tokenizer for testing
class MockLLMTokenizer
  attr_reader :pad_token_id, :eos_token_id

  def initialize
    @pad_token_id = 0
    @eos_token_id = 1
  end

  def encode(text, **_kwargs)
    tokens = text.split.take(20).map { |w| w.hash.abs % 1000 }
    {
      input_ids: [tokens]
    }
  end

  def decode(token_ids)
    "Decoded text for #{token_ids.size} tokens"
  end

  def vocab_size
    32000
  end
end

mock_tokenizer = MockLLMTokenizer.new

dataset = Fine::Datasets::InstructionDataset.from_jsonl(
  fixtures_path,
  tokenizer: mock_tokenizer,
  format: :alpaca,
  max_length: 128
)
puts "   Dataset size: #{dataset.size}"

puts "\n3. Testing data item..."
item = dataset[0]
puts "   Item keys: #{item.keys.join(', ')}"
puts "   Input IDs shape: #{item[:input_ids].shape.inspect}"
puts "   Labels shape: #{item[:labels].shape.inspect}"
puts "   Attention mask shape: #{item[:attention_mask].shape.inspect}"

puts "\n4. Testing InstructionDataLoader..."
loader = Fine::Datasets::InstructionDataLoader.new(dataset, batch_size: 2, shuffle: false)
batch = loader.first
puts "   Batch input_ids shape: #{batch[:input_ids].shape.inspect}"
puts "   Batch labels shape: #{batch[:labels].shape.inspect}"

puts "\n5. Testing LLM model components..."

# Test RMSNorm
puts "   Testing RMSNorm..."
norm = Fine::Models::RMSNorm.new(64)
test_input = Torch.randn([2, 10, 64])
norm_output = norm.call(test_input)
puts "   RMSNorm output shape: #{norm_output.shape.inspect}"

# Test LlamaMLP
puts "   Testing LlamaMLP..."
mlp = Fine::Models::LlamaMLP.new(hidden_size: 64, intermediate_size: 128)
mlp_output = mlp.call(test_input)
puts "   LlamaMLP output shape: #{mlp_output.shape.inspect}"

# Test RotaryEmbedding
puts "   Testing RotaryEmbedding..."
rope = Fine::Models::RotaryEmbedding.new(32, 128, 10000.0)
x = Torch.randn([2, 4, 10, 32])
position_ids = Torch.arange(10).unsqueeze(0).expand(2, -1)
cos, sin = rope.call(x, position_ids)
puts "   RoPE cos shape: #{cos.shape.inspect}"
puts "   RoPE sin shape: #{sin.shape.inspect}"

puts "\n" + "=" * 50
puts "LLM component tests passed!"
