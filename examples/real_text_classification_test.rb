#!/usr/bin/env ruby
# frozen_string_literal: true

# Test real text classification fine-tuning

require "bundler/setup"
require "fine"

puts "=" * 60
puts "TEXT CLASSIFICATION FINE-TUNING TEST"
puts "=" * 60

Fine.configure do |config|
  config.progress_bar = false
end

fixtures_path = File.expand_path("../spec/fixtures/text/reviews.jsonl", __dir__)

puts "\n1. Setting up TextClassifier with DistilBERT..."
classifier = Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  config.epochs = 2
  config.batch_size = 4
  config.learning_rate = 5e-5
  config.max_length = 128
end

puts "   Epochs: #{classifier.config.epochs}"
puts "   Batch size: #{classifier.config.batch_size}"
puts "   Max length: #{classifier.config.max_length}"

puts "\n2. Starting training on #{fixtures_path}..."
puts "   (This will download DistilBERT from HuggingFace if not cached)"

begin
  history = classifier.fit(train_file: fixtures_path)

  puts "\n3. Training completed!"
  puts "   Training history:"
  history.each_with_index do |metrics, i|
    puts "   Epoch #{i + 1}: loss=#{metrics[:loss].round(4)}, acc=#{(metrics[:accuracy] * 100).round(1)}%"
  end

  # Check if loss decreased
  if history.size >= 2
    if history.last[:loss] < history.first[:loss]
      puts "\n   ✓ Loss decreased from #{history.first[:loss].round(4)} to #{history.last[:loss].round(4)}"
    else
      puts "\n   ⚠ Loss did not decrease (may need more epochs or data)"
    end
  end

  puts "\n4. Testing predictions..."
  test_texts = [
    "This product is amazing and works perfectly!",
    "Terrible quality, broke after one day.",
    "It's okay, nothing special.",
    "Best purchase I've ever made!"
  ]

  predictions = classifier.predict(test_texts)
  test_texts.each_with_index do |text, i|
    pred = predictions[i].first
    puts "   \"#{text[0..40]}...\""
    puts "     → #{pred[:label]} (#{(pred[:score] * 100).round(1)}%)"
  end

  puts "\n5. Saving model..."
  save_path = "/tmp/fine_text_classifier"
  classifier.save(save_path)
  puts "   Saved to: #{save_path}"

  puts "\n6. Loading and re-testing..."
  loaded = Fine::TextClassifier.load(save_path)
  loaded_predictions = loaded.predict(test_texts.first)
  puts "   Loaded model prediction for first text:"
  puts "     → #{loaded_predictions.first.first[:label]} (#{(loaded_predictions.first.first[:score] * 100).round(1)}%)"

  puts "\n" + "=" * 60
  puts "TEXT CLASSIFICATION TEST PASSED!"
  puts "=" * 60

rescue => e
  puts "\n" + "=" * 60
  puts "TEXT CLASSIFICATION TEST FAILED!"
  puts "=" * 60
  puts "\nError: #{e.class}: #{e.message}"
  puts "\nBacktrace:"
  puts e.backtrace.first(15).join("\n")
  exit 1
end
