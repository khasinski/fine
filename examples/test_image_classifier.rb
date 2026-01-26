#!/usr/bin/env ruby
# frozen_string_literal: true

# Quick test of image classification with tiny local model

require "bundler/setup"
require "fine"

puts "Testing Fine::ImageClassifier..."
puts "=" * 50

# Use test fixtures
fixtures_path = File.expand_path("../spec/fixtures/images", __dir__)

puts "\n1. Creating ImageClassifier..."
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  config.epochs = 1
  config.batch_size = 2
  config.learning_rate = 1e-4
  config.image_size = 32  # Small for testing
  config.freeze_encoder = true  # Faster training
end

puts "   Config: epochs=#{classifier.config.epochs}, batch_size=#{classifier.config.batch_size}"

puts "\n2. Loading dataset from #{fixtures_path}..."
# Just test the dataset loading part without actual training
# (training requires downloading the model which takes time)

transforms = Fine::Transforms::Compose.new([
  Fine::Transforms::Resize.new(32),
  Fine::Transforms::ToTensor.new,
  Fine::Transforms::Normalize.new
])

dataset = Fine::Datasets::ImageDataset.from_directory(fixtures_path, transforms: transforms)
puts "   Dataset size: #{dataset.size}"
puts "   Classes: #{dataset.class_names.join(', ')}"
puts "   Label map: #{dataset.label_map}"

puts "\n3. Testing data loading..."
item = dataset[0]
puts "   Item keys: #{item.keys.join(', ')}"
puts "   Pixel values shape: #{item[:pixel_values].shape.inspect}"
puts "   Label: #{item[:label]}"

puts "\n4. Testing DataLoader..."
loader = Fine::Datasets::DataLoader.new(dataset, batch_size: 2, shuffle: true)
batch = loader.first
puts "   Batch pixel_values shape: #{batch[:pixel_values].shape.inspect}"
puts "   Batch labels: #{batch[:labels].to_a}"

puts "\n" + "=" * 50
puts "Basic tests passed!"
puts "\nNote: Full training requires downloading model weights from HuggingFace."
puts "Run with DOWNLOAD_MODELS=1 to test full training."

if ENV["DOWNLOAD_MODELS"]
  puts "\n" + "=" * 50
  puts "Downloading model and running full training..."

  begin
    classifier.fit(train_dir: fixtures_path, epochs: 1)
    puts "Training completed!"

    # Test save
    model_path = "/tmp/fine_test_model"
    classifier.save(model_path)
    puts "Model saved to #{model_path}"

    # Test load
    loaded = Fine::ImageClassifier.load(model_path)
    puts "Model loaded successfully!"

  rescue => e
    puts "Training failed: #{e.message}"
    puts e.backtrace.first(5).join("\n")
  end
end
