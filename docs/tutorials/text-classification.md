# Fine-tuning Text Classification Models

Classify text into categories—sentiment, spam, intent, topics.

## When to Use This

- Sentiment analysis (positive/negative/neutral reviews)
- Spam detection
- Intent classification for chatbots
- Support ticket routing
- Content moderation
- Topic categorization

## Supported Models

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| `distilbert-base-uncased` | 66M | Fast | Good |
| `bert-base-uncased` | 110M | Medium | Better |
| `roberta-base` | 125M | Medium | Better |
| `microsoft/deberta-v3-small` | 44M | Fast | Great |
| `microsoft/deberta-v3-base` | 86M | Medium | Best |

## Dataset Format

JSONL file with `text` and `label` fields:

```
data/train.jsonl
{"text": "This product exceeded my expectations!", "label": "positive"}
{"text": "Terrible quality, broke after one day", "label": "negative"}
{"text": "It's okay, nothing special", "label": "neutral"}
```

Or CSV:

```
data/train.csv
text,label
"This product exceeded my expectations!",positive
"Terrible quality, broke after one day",negative
```

## Basic Training

```ruby
require "fine"

classifier = Fine::TextClassifier.new("distilbert-base-uncased")

classifier.fit(
  train_file: "data/train.jsonl",
  val_file: "data/val.jsonl",
  epochs: 3
)

classifier.save("models/sentiment")
```

## Making Predictions

```ruby
classifier = Fine::TextClassifier.load("models/sentiment")

# Single text
result = classifier.predict("I love this product!")
puts result.first[:label]  # => "positive"
puts result.first[:score]  # => 0.97

# Batch prediction
results = classifier.predict([
  "Great service, highly recommend",
  "Worst purchase ever",
  "It works as described"
])
```

## Configuration

```ruby
classifier = Fine::TextClassifier.new("microsoft/deberta-v3-small") do |config|
  config.epochs = 5
  config.batch_size = 16
  config.learning_rate = 2e-5      # Lower than vision models
  config.max_length = 256          # Max tokens per text
  config.warmup_ratio = 0.1        # 10% of steps for warmup

  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch}: val_accuracy=#{(metrics[:val_accuracy] * 100).round(1)}%"
  end
end
```

## Use Cases

### Sentiment Analysis

```ruby
# Train on product reviews
classifier = Fine::TextClassifier.new("distilbert-base-uncased")
classifier.fit(train_file: "reviews.jsonl", epochs: 3)

# Analyze new reviews
def analyze_sentiment(review_text)
  result = classifier.predict(review_text).first
  {
    sentiment: result[:label],
    confidence: result[:score],
    needs_attention: result[:label] == "negative" && result[:score] > 0.8
  }
end
```

### Spam Detection

```ruby
classifier = Fine::TextClassifier.new("distilbert-base-uncased")
classifier.fit(train_file: "spam_ham.jsonl", epochs: 3)

def is_spam?(message)
  result = classifier.predict(message).first
  result[:label] == "spam" && result[:score] > 0.7
end
```

### Intent Classification

```ruby
# For chatbot / support routing
# Labels: billing, technical, shipping, general, cancel

classifier = Fine::TextClassifier.new("microsoft/deberta-v3-small")
classifier.fit(train_file: "support_intents.jsonl", epochs: 5)

def route_ticket(message)
  result = classifier.predict(message).first

  case result[:label]
  when "billing"
    assign_to_billing_team(message)
  when "cancel"
    assign_to_retention_team(message)
  when "technical"
    assign_to_tech_support(message)
  else
    assign_to_general_queue(message)
  end
end
```

### Multi-label Classification

For texts that can have multiple labels:

```ruby
classifier = Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  config.multi_label = true
  config.threshold = 0.5  # Predict labels above this confidence
end

# Data format for multi-label
# {"text": "Server crashed and lost data", "labels": ["technical", "urgent", "data_loss"]}

classifier.fit(train_file: "tickets_multilabel.jsonl")

result = classifier.predict("Payment failed and I can't login")
# => [{ label: "billing", score: 0.89 }, { label: "technical", score: 0.72 }]
```

## Data Preparation Tips

### Minimum Data

- 100+ examples per class for decent results
- 500+ examples per class for good results
- Balance classes or use class weights

### Class Imbalance

```ruby
classifier = Fine::TextClassifier.new("distilbert-base-uncased") do |config|
  config.class_weights = :balanced  # Auto-compute from data
  # Or manually: config.class_weights = { "positive" => 1.0, "negative" => 2.5 }
end
```

### Text Preprocessing

The tokenizer handles most preprocessing, but you might want to:

```ruby
def clean_text(text)
  text
    .gsub(/<[^>]+>/, ' ')     # Remove HTML
    .gsub(/https?:\S+/, '')   # Remove URLs
    .gsub(/\s+/, ' ')         # Normalize whitespace
    .strip
end
```

## Evaluation

```ruby
# Load test set
test_data = File.readlines("data/test.jsonl").map { |l| JSON.parse(l) }

predictions = classifier.predict(test_data.map { |d| d["text"] })
actuals = test_data.map { |d| d["label"] }

# Calculate accuracy
correct = predictions.zip(actuals).count { |pred, actual| pred.first[:label] == actual }
accuracy = correct.to_f / predictions.size
puts "Test accuracy: #{(accuracy * 100).round(1)}%"
```

## Troubleshooting

**Low accuracy:**
- Add more training data
- Check for mislabeled examples
- Try a larger model (deberta-v3-base)
- Increase epochs

**Overfitting:**
- Add more data
- Use dropout (increase in config)
- Reduce epochs
- Use a smaller model

**Slow training:**
- Reduce `max_length`
- Use a smaller model (distilbert)
- Reduce batch size if memory-constrained
