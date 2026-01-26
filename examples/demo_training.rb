#!/usr/bin/env ruby
# frozen_string_literal: true

# Demo: Fine-tune SigLIP2 for image classification
# This shows the full workflow with actual weight updates

require "bundler/setup"
require "fine"

puts "=" * 60
puts "FINE-TUNING DEMO"
puts "=" * 60

Fine.configure do |config|
  config.progress_bar = false
end

fixtures_path = File.expand_path("../spec/fixtures/images", __dir__)

puts "\n[1] Create and configure classifier"
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  config.epochs = 5
  config.batch_size = 2
  config.learning_rate = 1e-3  # Higher LR for faster learning on tiny dataset
  config.freeze_encoder = true  # Train only classification head

  config.on_epoch_end do |epoch, metrics|
    puts "    Epoch #{epoch + 1}: loss=#{metrics[:loss].round(4)}, acc=#{(metrics[:accuracy] * 100).round(1)}%"
  end
end

puts "\n[2] Train on #{fixtures_path}"
puts "    (4 images: 2 cats, 2 dogs)"
puts ""

history = classifier.fit(train_dir: fixtures_path)

puts "\n[3] Training metrics"
initial_loss = history.first[:loss]
final_loss = history.last[:loss]
initial_acc = history.first[:accuracy]
final_acc = history.last[:accuracy]

puts "    Initial: loss=#{initial_loss.round(4)}, accuracy=#{(initial_acc * 100).round(1)}%"
puts "    Final:   loss=#{final_loss.round(4)}, accuracy=#{(final_acc * 100).round(1)}%"

if final_loss < initial_loss
  puts "    ✓ Loss decreased by #{((1 - final_loss/initial_loss) * 100).round(1)}%"
else
  puts "    ⚠ Loss did not decrease"
end

if final_acc > initial_acc
  puts "    ✓ Accuracy improved from #{(initial_acc * 100).round(1)}% to #{(final_acc * 100).round(1)}%"
end

puts "\n[4] Test predictions"
Dir.glob(File.join(fixtures_path, "*/*.jpg")).each do |image_path|
  true_label = File.basename(File.dirname(image_path))
  predictions = classifier.predict(image_path).first
  pred_label = predictions.first[:label]
  pred_score = predictions.first[:score]

  status = pred_label == true_label ? "✓" : "✗"
  puts "    #{status} #{File.basename(image_path)}: predicted=#{pred_label} (#{(pred_score * 100).round(1)}%), actual=#{true_label}"
end

puts "\n[5] Save and reload model"
save_path = "/tmp/fine_demo_model"
classifier.save(save_path)
puts "    Saved to: #{save_path}"

loaded = Fine::ImageClassifier.load(save_path)
puts "    Loaded successfully with #{loaded.label_map.size} classes: #{loaded.label_map.keys.join(', ')}"

puts "\n" + "=" * 60
puts "DEMO COMPLETE"
puts "=" * 60
