#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Fine-tune an image classifier for shape/color recognition
#
# This example demonstrates how to fine-tune SigLIP2 for classifying
# images by dominant color (which corresponds to different shapes).

require "bundler/setup"
require "fine"

puts "=" * 60
puts "SHAPE CLASSIFICATION EXAMPLE"
puts "=" * 60

Fine.configure { |c| c.progress_bar = false }

data_dir = File.expand_path("data/shapes", __dir__)
save_path = "/tmp/shape-classifier"

# Check if images exist
unless File.exist?(data_dir) && Dir.glob("#{data_dir}/*/*.jpg").any?
  puts "Generating test images..."
  require_relative "generate_shape_images"
end

puts "\n1. Creating classifier with SigLIP2..."

classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  config.epochs = 5
  config.batch_size = 4
  config.learning_rate = 1e-4
  config.image_size = 224
  config.freeze_encoder = false  # Fine-tune the whole model

  config.on_epoch_end do |epoch, metrics|
    puts "   Epoch #{epoch}: loss=#{metrics[:loss].round(4)}"
  end
end

# Count images
image_count = Dir.glob("#{data_dir}/*/*.jpg").count
class_count = Dir.glob("#{data_dir}/*").select { |f| File.directory?(f) }.count

puts "\n2. Fine-tuning on #{data_dir}..."
puts "   (#{image_count} images across #{class_count} classes)"
puts "   Classes: #{Dir.glob("#{data_dir}/*").select { |f| File.directory?(f) }.map { |f| File.basename(f) }.join(", ")}"

history = classifier.fit(train_dir: data_dir, epochs: 5)

puts "\n3. Training complete!"
puts "   Initial loss: #{history.first[:loss].round(4)}"
puts "   Final loss: #{history.last[:loss].round(4)}"

improvement = ((1 - history.last[:loss] / history.first[:loss]) * 100).round(1)
puts "   Improvement: #{improvement}%"

puts "\n4. Testing predictions on training images..."

# Test on one image from each class
%w[circle square triangle].each do |shape|
  test_image = Dir.glob("#{data_dir}/#{shape}/*.jpg").first
  predictions = classifier.predict(test_image, top_k: 3)
  top = predictions.first.first

  puts "   #{shape}_1.jpg => #{top[:label]} (#{(top[:score] * 100).round(1)}%)"
end

puts "\n5. Saving model to #{save_path}..."
classifier.save(save_path)

puts "\n6. Loading and verifying saved model..."
loaded = Fine::ImageClassifier.load(save_path)

test_image = Dir.glob("#{data_dir}/circle/*.jpg").first
original_pred = classifier.predict(test_image).first.first
loaded_pred = loaded.predict(test_image).first.first

puts "   Original: #{original_pred[:label]} (#{original_pred[:score].round(4)})"
puts "   Loaded:   #{loaded_pred[:label]} (#{loaded_pred[:score].round(4)})"

puts "\n" + "=" * 60
puts "SHAPE CLASSIFICATION COMPLETE!"
puts "=" * 60
puts "\nModel saved to: #{save_path}"
puts "Load with: Fine::ImageClassifier.load('#{save_path}')"
puts "Classes: #{classifier.class_names.join(", ")}"
