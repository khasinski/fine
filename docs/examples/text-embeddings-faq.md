# Text Embeddings: Customer Support FAQ Matching

This example demonstrates fine-tuning a sentence transformer for semantic search in a customer support context.

## Setup

```ruby
require "fine"

Fine.configure { |c| c.progress_bar = false }
```

## Training Data Format

Create query-answer pairs in JSONL format:

```jsonl
{"query": "How do I reset my password?", "positive": "To reset your password, click 'Forgot Password' on the login page."}
{"query": "I forgot my login credentials", "positive": "To reset your password, click 'Forgot Password' on the login page."}
{"query": "How long does shipping take?", "positive": "Standard shipping takes 3-5 business days."}
```

Multiple queries can map to the same answer to teach semantic similarity.

## Train the Embedder

```ruby
embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2") do |config|
  config.epochs = 2
  config.batch_size = 8
  config.learning_rate = 2e-5
end

# Test pre-training similarity
pre_sim = embedder.similarity("How can I get my money back?", "To initiate a return, go to your orders...")
puts "Pre-training: #{pre_sim.round(4)}"  # => 0.4008

# Fine-tune
history = embedder.fit(train_file: "data/support_faq_pairs.jsonl")

# Test post-training
post_sim = embedder.similarity("How can I get my money back?", "To initiate a return, go to your orders...")
puts "Post-training: #{post_sim.round(4)}"  # => 0.4723
```

## Training Results

```
Epoch 0: loss=4.3761
Epoch 1: loss=5.8889

Post-training similarity: 0.4723
Improvement: 7.15 percentage points
```

## Semantic Search

```ruby
faq_corpus = [
  "To reset your password, click 'Forgot Password' on the login page.",
  "Standard shipping takes 3-5 business days.",
  "To initiate a return, go to your order history and click 'Request Return'.",
  "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
  "Yes, we ship to over 50 countries.",
  "You can reach us via live chat, email, or phone at 1-800-555-0123."
]

# Search for relevant FAQ
results = embedder.search("I need to get my money back", faq_corpus, top_k: 2)
# => [
#      { text: "To initiate a return...", score: 0.548 },
#      { text: "You can reach us via...", score: 0.425 }
#    ]

results = embedder.search("What's the phone number for help?", faq_corpus)
# => [{ text: "You can reach us via...", score: 0.461 }]

results = embedder.search("Can you deliver to Germany?", faq_corpus)
# => [{ text: "Yes, we ship to over 50 countries...", score: 0.515 }]
```

## Save and Load

```ruby
# Save
embedder.save("/tmp/support-faq-embedder")

# Load later
loaded = Fine::TextEmbedder.load("/tmp/support-faq-embedder")
loaded.search("your query", corpus)
```

## Key Takeaways

- Even 2 epochs of fine-tuning improves similarity for domain-specific queries
- Multiple Negatives Ranking Loss learns to distinguish relevant from irrelevant answers
- The model correctly identifies the best FAQ match for natural language queries
- Pre-trained models already provide good baseline (0.4 similarity), fine-tuning improves it
