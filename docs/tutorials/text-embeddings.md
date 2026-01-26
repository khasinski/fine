# Fine-tuning Text Embedding Models

Train text embeddings for semantic search, clustering, and similarity matching.

## When to Use This

- Build semantic search for your domain (legal docs, medical records, code)
- Improve retrieval for RAG applications
- Cluster similar documents
- Match queries to FAQ answers
- Detect duplicate content

## Supported Models

| Model | Size | Use Case |
|-------|------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Fast, general purpose |
| `sentence-transformers/all-mpnet-base-v2` | 110M | Better quality |
| `BAAI/bge-small-en-v1.5` | 33M | Retrieval optimized |
| `BAAI/bge-base-en-v1.5` | 110M | Best retrieval quality |

## Dataset Format

Prepare your data as pairs or triplets:

### Pairs (query, positive)

```
data/pairs.jsonl
{"query": "How do I reset my password?", "positive": "To reset your password, click 'Forgot Password' on the login page."}
{"query": "What's your return policy?", "positive": "We accept returns within 30 days of purchase."}
```

### Triplets (query, positive, negative)

```
data/triplets.jsonl
{"query": "python list append", "positive": "list.append(x) adds item x to the end", "negative": "list.pop() removes the last item"}
```

## Basic Training

```ruby
require "fine"

embedder = Fine::TextEmbedder.new("sentence-transformers/all-MiniLM-L6-v2")

embedder.fit(
  train_file: "data/pairs.jsonl",
  epochs: 3
)

embedder.save("models/my_embedder")
```

## Training for Semantic Search

Optimize for retrieval with contrastive loss:

```ruby
embedder = Fine::TextEmbedder.new("BAAI/bge-base-en-v1.5") do |config|
  config.epochs = 5
  config.batch_size = 32
  config.learning_rate = 2e-5
  config.loss = :multiple_negatives_ranking  # Best for retrieval

  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch}: loss=#{metrics[:loss]}"
  end
end

embedder.fit(train_file: "data/queries_documents.jsonl")
embedder.save("models/search_embedder")
```

## Generating Embeddings

```ruby
embedder = Fine::TextEmbedder.load("models/my_embedder")

# Single text
embedding = embedder.encode("How do I cancel my subscription?")
# => Array of 384 floats (for MiniLM)

# Batch encoding
embeddings = embedder.encode([
  "First document",
  "Second document",
  "Third document"
])
# => Array of 3 embeddings
```

## Building a Search Index

```ruby
class SemanticSearch
  def initialize(embedder)
    @embedder = embedder
    @documents = []
    @embeddings = []
  end

  def add(text, metadata = {})
    @documents << { text: text, metadata: metadata }
    @embeddings << @embedder.encode(text)
  end

  def search(query, top_k: 5)
    query_emb = @embedder.encode(query)

    scores = @embeddings.map { |emb| cosine_similarity(query_emb, emb) }

    scores
      .each_with_index
      .sort_by { |score, _| -score }
      .first(top_k)
      .map { |score, idx| { document: @documents[idx], score: score } }
  end

  private

  def cosine_similarity(a, b)
    dot = a.zip(b).sum { |x, y| x * y }
    norm_a = Math.sqrt(a.sum { |x| x * x })
    norm_b = Math.sqrt(b.sum { |x| x * x })
    dot / (norm_a * norm_b)
  end
end

# Usage
search = SemanticSearch.new(embedder)

# Index your documents
documents.each { |doc| search.add(doc[:text], id: doc[:id]) }

# Search
results = search.search("how to get a refund")
results.each do |r|
  puts "#{r[:score].round(3)}: #{r[:document][:text][0..50]}..."
end
```

## Domain Adaptation

Fine-tune on your specific domain for better results:

```ruby
# Prepare domain-specific pairs
# Example: customer support queries matched to answers
pairs = [
  { query: "broken screen", positive: "Screen repair costs $99 and takes 2-3 days" },
  { query: "cracked display", positive: "Screen repair costs $99 and takes 2-3 days" },
  # ... more domain-specific examples
]

# Save as JSONL
File.write("data/support_pairs.jsonl", pairs.map(&:to_json).join("\n"))

# Fine-tune
embedder = Fine::TextEmbedder.new("sentence-transformers/all-mpnet-base-v2")
embedder.fit(train_file: "data/support_pairs.jsonl", epochs: 3)
embedder.save("models/support_search")
```

## Hard Negatives Mining

For better quality, use hard negatives (similar but wrong answers):

```ruby
embedder = Fine::TextEmbedder.new("BAAI/bge-base-en-v1.5") do |config|
  config.loss = :triplet
  config.margin = 0.5  # Minimum distance between positive and negative
end

# Triplet format
embedder.fit(train_file: "data/triplets.jsonl")
```

## Evaluation

Test retrieval quality:

```ruby
def evaluate_retrieval(embedder, test_queries, ground_truth)
  hits_at_1 = 0
  hits_at_5 = 0

  test_queries.each do |query, expected_doc_id|
    results = search.search(query, top_k: 5)

    result_ids = results.map { |r| r[:document][:metadata][:id] }

    hits_at_1 += 1 if result_ids[0] == expected_doc_id
    hits_at_5 += 1 if result_ids.include?(expected_doc_id)
  end

  {
    recall_at_1: hits_at_1.to_f / test_queries.size,
    recall_at_5: hits_at_5.to_f / test_queries.size
  }
end
```

## Best Practices

1. **Start with a good base model** - BGE models work best for retrieval
2. **Use domain-specific data** - Even 1000 pairs helps significantly
3. **Include hard negatives** - Similar but incorrect matches improve precision
4. **Evaluate on held-out queries** - Don't overfit to training data
5. **Consider asymmetric models** - Some models use different encodings for queries vs documents
