#!/usr/bin/env ruby
# frozen_string_literal: true

# Basic image classification example
#
# This example shows how to fine-tune a SigLIP2 model for image classification.
#
# Dataset structure expected:
#   data/
#     train/
#       cat/
#         cat1.jpg
#         cat2.jpg
#       dog/
#         dog1.jpg
#         dog2.jpg
#     val/
#       cat/
#         cat3.jpg
#       dog/
#         dog3.jpg

require "fine"

# Configure Fine (optional)
Fine.configure do |config|
  config.cache_dir = File.expand_path("~/.cache/fine")
  config.progress_bar = true
end

# Create a classifier
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  # Training settings
  config.epochs = 3
  config.batch_size = 8
  config.learning_rate = 2e-4

  # Model settings
  config.freeze_encoder = false  # Full fine-tuning
  config.dropout = 0.1

  # Callbacks
  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch + 1}:"
    puts "  Train Loss: #{metrics[:loss].round(4)}"
    puts "  Train Acc:  #{(metrics[:accuracy] * 100).round(2)}%"
    if metrics[:val_loss]
      puts "  Val Loss:   #{metrics[:val_loss].round(4)}"
      puts "  Val Acc:    #{(metrics[:val_accuracy] * 100).round(2)}%"
    end
  end
end

# Train the model
puts "Starting training..."
history = classifier.fit(
  train_dir: "data/train",
  val_dir: "data/val"
)

# Save the model
classifier.save("models/my_classifier")
puts "Model saved to models/my_classifier"

# Make predictions
puts "\nMaking predictions..."
predictions = classifier.predict("data/test/image.jpg")
predictions.each do |pred|
  puts "  #{pred[:label]}: #{(pred[:score] * 100).round(2)}%"
end
