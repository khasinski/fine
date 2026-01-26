#!/usr/bin/env ruby
# frozen_string_literal: true

# Test text embeddings (without fine-tuning, which requires more data)

require "bundler/setup"
require "fine"

puts "=" * 60
puts "TEXT EMBEDDER TEST"
puts "=" * 60

Fine.configure do |config|
  config.progress_bar = false
end

puts "\n1. Loading SentenceTransformer (all-MiniLM-L6-v2)..."
puts "   (This will download from HuggingFace if not cached)"

begin
  embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2") do |config|
    config.max_length = 128
  end

  puts "   Model loaded successfully!"
  puts "   Embedding dimension: #{embedder.embedding_dim}"

  puts "\n2. Testing single text encoding..."
  text = "The quick brown fox jumps over the lazy dog."
  embedding = embedder.encode(text)
  puts "   Text: \"#{text}\""
  puts "   Embedding shape: [#{embedding.size}]"
  puts "   First 5 values: #{embedding.first(5).map { |v| v.round(4) }}"

  puts "\n3. Testing batch encoding..."
  texts = [
    "I love machine learning!",
    "Deep learning is fascinating.",
    "The weather is nice today.",
    "Ruby is a great programming language."
  ]
  embeddings = embedder.encode(texts)
  puts "   Encoded #{texts.size} texts"
  puts "   Result shapes: #{embeddings.size} x #{embeddings.first.size}"

  puts "\n4. Testing semantic similarity..."
  pairs = [
    ["I love programming", "Coding is my passion"],
    ["I love programming", "The sky is blue"],
    ["Machine learning is cool", "AI and ML are interesting"],
    ["Dogs are pets", "Cats are animals"]
  ]

  pairs.each do |text_a, text_b|
    similarity = embedder.similarity(text_a, text_b)
    puts "   \"#{text_a[0..25]}...\" vs \"#{text_b[0..25]}...\""
    puts "     → Similarity: #{(similarity * 100).round(1)}%"
  end

  puts "\n5. Testing semantic search..."
  query = "machine learning"
  corpus = [
    "I love pizza",
    "Deep learning is a subset of machine learning",
    "The stock market is volatile",
    "Neural networks can learn complex patterns",
    "Ruby on Rails is a web framework",
    "Artificial intelligence is transforming industries"
  ]

  results = embedder.search(query, corpus, top_k: 3)
  puts "   Query: \"#{query}\""
  puts "   Top 3 results:"
  results.each_with_index do |result, i|
    puts "     #{i + 1}. \"#{result[:text][0..45]}...\" (#{(result[:score] * 100).round(1)}%)"
  end

  puts "\n6. Saving and loading model..."
  save_path = "/tmp/fine_text_embedder"
  embedder.save(save_path)
  puts "   Saved to: #{save_path}"

  loaded = Fine::TextEmbedder.load(save_path)
  puts "   Loaded successfully!"

  # Verify embeddings match
  original_emb = embedder.encode("test text")
  loaded_emb = loaded.encode("test text")
  diff = original_emb.zip(loaded_emb).map { |a, b| (a - b).abs }.max
  puts "   Max embedding difference: #{diff.round(6)}"

  if diff < 0.0001
    puts "   ✓ Embeddings match!"
  else
    puts "   ⚠ Embeddings differ"
  end

  puts "\n" + "=" * 60
  puts "TEXT EMBEDDER TEST PASSED!"
  puts "=" * 60

rescue => e
  puts "\n" + "=" * 60
  puts "TEXT EMBEDDER TEST FAILED!"
  puts "=" * 60
  puts "\nError: #{e.class}: #{e.message}"
  puts "\nBacktrace:"
  puts e.backtrace.first(15).join("\n")
  exit 1
end
