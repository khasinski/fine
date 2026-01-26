#!/usr/bin/env ruby
# frozen_string_literal: true

# Example: Fine-tune text embeddings for customer support FAQ matching
#
# This example demonstrates how to fine-tune a sentence transformer model
# for semantic search in a customer support context.

require "bundler/setup"
require "fine"

puts "=" * 60
puts "SUPPORT FAQ EMBEDDINGS EXAMPLE"
puts "=" * 60

Fine.configure { |c| c.progress_bar = false }

data_path = File.expand_path("data/support_faq_pairs.jsonl", __dir__)
save_path = "/tmp/support-faq-embedder"

puts "\n1. Creating embedder with all-MiniLM-L6-v2..."

embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2") do |config|
  config.epochs = 2
  config.batch_size = 8
  config.learning_rate = 2e-5

  config.on_epoch_end do |epoch, metrics|
    puts "   Epoch #{epoch}: loss=#{metrics[:loss].round(4)}"
  end
end

puts "   Embedding dimension: #{embedder.embedding_dim}"

# Test pre-training similarity
puts "\n2. Testing pre-training similarity..."
test_query = "How can I get my money back?"
faq_answer = "To initiate a return, go to your order history and click 'Request Return' within 30 days of delivery."

pre_similarity = embedder.similarity(test_query, faq_answer)
puts "   Query: \"#{test_query}\""
puts "   FAQ: \"#{faq_answer[0, 60]}...\""
puts "   Pre-training similarity: #{pre_similarity.round(4)}"

puts "\n3. Fine-tuning on #{data_path}..."
puts "   (#{File.readlines(data_path).count} query-answer pairs)"

history = embedder.fit(train_file: data_path, epochs: 2)

puts "\n4. Training complete!"
puts "   Initial loss: #{history.first[:loss].round(4)}"
puts "   Final loss: #{history.last[:loss].round(4)}"

# Test post-training similarity
puts "\n5. Testing post-training similarity..."
post_similarity = embedder.similarity(test_query, faq_answer)
puts "   Post-training similarity: #{post_similarity.round(4)}"
puts "   Improvement: #{((post_similarity - pre_similarity) * 100).round(2)} percentage points"

# Semantic search demo
puts "\n6. Semantic search demo..."

faq_corpus = [
  "To reset your password, click 'Forgot Password' on the login page and enter your email address.",
  "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
  "To initiate a return, go to your order history and click 'Request Return' within 30 days of delivery.",
  "You can cancel your order within 1 hour of placing it. Go to your orders and click 'Cancel Order'.",
  "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
  "Once shipped, you'll receive a tracking number via email. Click the link to see real-time updates.",
  "Yes, we ship to over 50 countries. International shipping takes 7-14 business days.",
  "You can reach us via live chat, email at support@example.com, or phone at 1-800-555-0123."
]

test_queries = [
  "I need to get my money back for this purchase",
  "What's the phone number for help?",
  "Can you deliver to Germany?"
]

test_queries.each do |query|
  results = embedder.search(query, faq_corpus, top_k: 2)
  puts "\n   Query: \"#{query}\""
  results.each_with_index do |result, i|
    puts "   #{i + 1}. (#{result[:score].round(3)}) #{result[:text][0, 60]}..."
  end
end

puts "\n7. Saving model to #{save_path}..."
embedder.save(save_path)

puts "\n8. Loading and verifying saved model..."
loaded = Fine::TextEmbedder.load(save_path)

original_emb = embedder.encode("test query")
loaded_emb = loaded.encode("test query")

# Check embeddings are the same
diff = original_emb.zip(loaded_emb).sum { |a, b| (a - b).abs }
puts "   Embedding difference: #{diff.round(6)} (should be ~0)"

puts "\n" + "=" * 60
puts "SUPPORT FAQ EMBEDDINGS COMPLETE!"
puts "=" * 60
puts "\nModel saved to: #{save_path}"
puts "Load with: Fine::TextEmbedder.load('#{save_path}')"
