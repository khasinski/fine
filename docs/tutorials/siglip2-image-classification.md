# Fine-tuning SigLIP2 for Image Classification

Classify images into predefined categories (e.g., cats vs dogs, product types, document categories).

## When to Use This

- You have images that belong to distinct categories
- You want to automatically sort or label images
- Examples: photo organization, content moderation, product categorization

## Dataset Structure

Organize images in folders named after each class:

```
data/
  train/
    cats/
      cat001.jpg
      cat002.jpg
      ...
    dogs/
      dog001.jpg
      dog002.jpg
      ...
  val/
    cats/
      cat101.jpg
    dogs/
      dog101.jpg
```

**Tips:**
- Aim for at least 20-50 images per class for decent results
- More images = better accuracy
- Balance classes roughly equally
- Use clear, representative images

## Basic Training

```ruby
require "fine"

classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224")

classifier.fit(
  train_dir: "data/train",
  val_dir: "data/val",
  epochs: 3
)

classifier.save("models/my_classifier")
```

## Training with Configuration

```ruby
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  # Training settings
  config.epochs = 5
  config.batch_size = 16          # Lower if running out of memory
  config.learning_rate = 2e-4     # Default works well for most cases

  # Freeze encoder for faster training (less accurate)
  config.freeze_encoder = false   # Set true for feature extraction only

  # Track progress
  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch + 1}: accuracy=#{(metrics[:accuracy] * 100).round(1)}%"
  end
end

classifier.fit(train_dir: "data/train", val_dir: "data/val")
```

## Making Predictions

```ruby
# Load trained model
classifier = Fine::ImageClassifier.load("models/my_classifier")

# Single image
results = classifier.predict("photo.jpg")
puts results.first[:label]  # => "cats"
puts results.first[:score]  # => 0.95

# Multiple images
results = classifier.predict(["photo1.jpg", "photo2.jpg", "photo3.jpg"])
results.each_with_index do |preds, i|
  puts "Image #{i + 1}: #{preds.first[:label]}"
end
```

## Using Without Validation Set

If you don't have a separate validation set:

```ruby
classifier.fit_with_split(
  data_dir: "data/all_images",
  val_split: 0.2,   # Use 20% for validation
  epochs: 3
)
```

## Model Selection

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `siglip2-base-patch16-224` | 86M | Fast | Good | Quick experiments, limited GPU |
| `siglip2-base-patch16-384` | 86M | Medium | Better | Production with good GPU |
| `siglip2-large-patch16-256` | 303M | Slower | Best | Maximum accuracy |

## Troubleshooting

**Out of memory:**
- Reduce `batch_size` (try 8 or 4)
- Use a smaller model variant
- Use `freeze_encoder = true`

**Low accuracy:**
- Add more training images
- Train for more epochs
- Check for mislabeled images
- Ensure classes are visually distinct

**Overfitting (train accuracy high, val accuracy low):**
- Add more training data
- Reduce epochs
- Use `freeze_encoder = true`
