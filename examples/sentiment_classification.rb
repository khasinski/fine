#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Fine-tune a sentiment classifier for product reviews
#
# This example demonstrates how to fine-tune DistilBERT for binary
# sentiment classification (positive/negative reviews).
#
# NOTE: Text classification is currently experimental. For best results:
# - Use larger datasets (100+ samples per class)
# - Train for more epochs (20+)
# - Consider using a pre-fine-tuned sentiment model

require "bundler/setup"
require "fine"

puts "=" * 60
puts "SENTIMENT CLASSIFICATION EXAMPLE"
puts "=" * 60

# Disable progress bar for cleaner output
Fine.configure { |c| c.progress_bar = false }

data_path = File.expand_path("data/sentiment_reviews.jsonl", __dir__)
save_path = "/tmp/sentiment-classifier"

puts "\n1. Creating classifier with distilbert-base-uncased..."

classifier = Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  config.epochs = 10
  config.batch_size = 8
  config.learning_rate = 5e-5  # Slightly higher for small dataset
  config.max_length = 128

  config.on_epoch_end do |epoch, metrics|
    acc_str = metrics[:accuracy] ? ", acc=#{(metrics[:accuracy] * 100).round(1)}%" : ""
    puts "   Epoch #{epoch}: loss=#{metrics[:loss].round(4)}#{acc_str}"
  end
end

puts "\n2. Fine-tuning on #{data_path}..."
puts "   (#{File.readlines(data_path).count} examples)"

history = classifier.fit(train_file: data_path, epochs: 10)

puts "\n3. Training complete!"
puts "   Initial loss: #{history.first[:loss].round(4)}"
puts "   Final loss: #{history.last[:loss].round(4)}"

improvement = ((1 - history.last[:loss] / history.first[:loss]) * 100).round(1)
puts "   Improvement: #{improvement}%"

puts "\n4. Testing predictions..."

test_samples = [
  "This is the best product I've ever purchased! Amazing quality.",
  "Terrible experience. Product arrived broken and support ignored me.",
  "Decent product for the price. Does what it's supposed to do.",
  "Complete waste of money. Returning immediately.",
  "Exceeded my expectations! Will buy again."
]

test_samples.each do |text|
  predictions = classifier.predict(text, top_k: 2)
  top = predictions.first.first
  puts "   \"#{text[0, 50]}...\""
  puts "   => #{top[:label]} (#{(top[:score] * 100).round(1)}%)\n\n"
end

puts "5. Saving model to #{save_path}..."
classifier.save(save_path)

puts "\n6. Loading and verifying saved model..."
loaded = Fine::TextClassifier.load(save_path)

test_text = "Outstanding quality and fast shipping!"
original_pred = classifier.predict(test_text).first.first
loaded_pred = loaded.predict(test_text).first.first

puts "   Original: #{original_pred[:label]} (#{original_pred[:score]})"
puts "   Loaded:   #{loaded_pred[:label]} (#{loaded_pred[:score]})"

puts "\n" + "=" * 60
puts "SENTIMENT CLASSIFICATION COMPLETE!"
puts "=" * 60
puts "\nModel saved to: #{save_path}"
puts "Load with: Fine::TextClassifier.load('#{save_path}')"
