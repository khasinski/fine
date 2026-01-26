#!/usr/bin/env ruby
# frozen_string_literal: true

# Quick test of text classification with local fixtures

require "bundler/setup"
require "fine"

puts "Testing Fine::TextClassifier..."
puts "=" * 50

# Use test fixtures
fixtures_path = File.expand_path("../spec/fixtures/text/reviews.jsonl", __dir__)

puts "\n1. Creating TextClassifier..."
classifier = Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  config.epochs = 1
  config.batch_size = 2
  config.learning_rate = 2e-5
  config.max_length = 64
end

puts "   Config: epochs=#{classifier.config.epochs}, batch_size=#{classifier.config.batch_size}, max_length=#{classifier.config.max_length}"

puts "\n2. Testing TextDataset loading..."

# Create a mock tokenizer for testing
class MockTokenizer
  def encode(texts, **_kwargs)
    texts = [texts] if texts.is_a?(String)
    {
      input_ids: texts.map { |_| (1..10).to_a },
      attention_mask: texts.map { |_| [1] * 10 },
      token_type_ids: texts.map { |_| [0] * 10 }
    }
  end
end

mock_tokenizer = MockTokenizer.new

dataset = Fine::Datasets::TextDataset.from_file(fixtures_path, tokenizer: mock_tokenizer)
puts "   Dataset size: #{dataset.size}"
puts "   Classes: #{dataset.num_classes}"
puts "   Label map: #{dataset.label_map}"

puts "\n3. Testing data item..."
item = dataset[0]
puts "   Item keys: #{item.keys.join(', ')}"
puts "   Input IDs length: #{item[:input_ids].size}"
puts "   Label: #{item[:label]}"

puts "\n4. Testing TextDataLoader..."
loader = Fine::Datasets::TextDataLoader.new(dataset, batch_size: 2, shuffle: false)
batch = loader.first
puts "   Batch input_ids shape: #{batch[:input_ids].shape.inspect}"
puts "   Batch labels: #{batch[:labels].to_a}"

puts "\n" + "=" * 50
puts "Text classification tests passed!"
