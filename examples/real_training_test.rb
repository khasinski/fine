#!/usr/bin/env ruby
# frozen_string_literal: true

# Actually fine-tune a model to verify training works

require "bundler/setup"
require "fine"

puts "=" * 60
puts "REAL FINE-TUNING TEST"
puts "=" * 60

# Disable progress bar for cleaner output
Fine.configure do |config|
  config.progress_bar = false
end

fixtures_path = File.expand_path("../spec/fixtures/images", __dir__)

puts "\n1. Setting up ImageClassifier with SigLIP2..."
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  config.epochs = 2
  config.batch_size = 2
  config.learning_rate = 1e-4
  config.image_size = 224
  config.freeze_encoder = true  # Only train classification head
end

puts "   Epochs: #{classifier.config.epochs}"
puts "   Batch size: #{classifier.config.batch_size}"
puts "   Learning rate: #{classifier.config.learning_rate}"

puts "\n2. Starting training on #{fixtures_path}..."
puts "   (This will download the model from HuggingFace if not cached)"

begin
  history = classifier.fit(train_dir: fixtures_path, epochs: 2)

  puts "\n3. Training completed!"
  puts "   Training history:"
  history.each do |epoch_data|
    puts "   Epoch #{epoch_data[:epoch]}: loss=#{epoch_data[:loss].round(4)}"
  end

  # Check if loss decreased
  if history.size >= 2
    if history.last[:loss] < history.first[:loss]
      puts "\n   ✓ Loss decreased from #{history.first[:loss].round(4)} to #{history.last[:loss].round(4)}"
    else
      puts "\n   ⚠ Loss did not decrease (may need more epochs or data)"
    end
  end

  puts "\n4. Testing prediction..."
  test_image = Dir.glob(File.join(fixtures_path, "*/*.jpg")).first
  predictions = classifier.predict(test_image)
  puts "   Image: #{File.basename(test_image)}"
  puts "   Predictions:"
  predictions.first.each do |pred|
    puts "     #{pred[:label]}: #{(pred[:score] * 100).round(1)}%"
  end

  puts "\n5. Saving model..."
  save_path = "/tmp/fine_trained_model"
  classifier.save(save_path)
  puts "   Saved to: #{save_path}"

  puts "\n6. Loading and re-testing..."
  loaded = Fine::ImageClassifier.load(save_path)
  loaded_predictions = loaded.predict(test_image)
  puts "   Loaded model predictions:"
  loaded_predictions.first.each do |pred|
    puts "     #{pred[:label]}: #{(pred[:score] * 100).round(1)}%"
  end

  puts "\n" + "=" * 60
  puts "FINE-TUNING TEST PASSED!"
  puts "=" * 60

rescue => e
  puts "\n" + "=" * 60
  puts "FINE-TUNING FAILED!"
  puts "=" * 60
  puts "\nError: #{e.class}: #{e.message}"
  puts "\nBacktrace:"
  puts e.backtrace.first(15).join("\n")
  exit 1
end
