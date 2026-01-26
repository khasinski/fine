# Fine-tuning SigLIP2 for New Object Recognition

Teach the model to recognize specific objects, products, or concepts it hasn't seen before.

## When to Use This

- You want to detect your specific products, logos, or items
- You need to recognize custom objects not in standard datasets
- Examples: your product catalog, brand logos, custom equipment, specific animals/plants

## How It Works

SigLIP2 learns visual concepts from image-text pairs. For object recognition, we fine-tune it to associate your specific objects with labels, enabling it to recognize them in new images.

## Dataset Structure

Create folders for each object you want to recognize:

```
data/
  train/
    my_product_a/
      product_a_01.jpg
      product_a_02.jpg
      product_a_angle1.jpg
      product_a_lighting2.jpg
      ...
    my_product_b/
      product_b_01.jpg
      ...
    background/           # Optional: negative examples
      random_scene_01.jpg
      ...
```

**Tips for Object Recognition:**
- Capture objects from multiple angles
- Vary lighting conditions
- Include different backgrounds
- Show objects at different scales
- 30-100 images per object works well

## Training

```ruby
require "fine"

# Use a larger model for better object recognition
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-384") do |config|
  config.epochs = 5
  config.batch_size = 16
  config.learning_rate = 1e-4     # Slightly lower for fine-grained recognition
  config.freeze_encoder = false   # Full fine-tuning for new concepts
end

classifier.fit(
  train_dir: "data/train",
  val_dir: "data/val"
)

classifier.save("models/product_recognizer")
```

## Recognizing Objects in New Images

```ruby
recognizer = Fine::ImageClassifier.load("models/product_recognizer")

# Check what object is in an image
results = recognizer.predict("customer_photo.jpg")

# Get top prediction
top = results.first
if top[:score] > 0.7
  puts "Detected: #{top[:label]} (#{(top[:score] * 100).round}% confident)"
else
  puts "No confident match found"
end

# See all possibilities
results.each do |pred|
  puts "#{pred[:label]}: #{(pred[:score] * 100).round}%"
end
```

## Including Negative Examples

For better precision, include a "background" or "other" class:

```
data/train/
  product_a/
  product_b/
  other/          # Images without your products
```

This helps the model learn what your objects are NOT.

## Multi-Object Detection Strategy

If images might contain multiple objects, train separate binary classifiers:

```ruby
# Train one model per object type
products = ["product_a", "product_b", "product_c"]

products.each do |product|
  classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224")

  # Binary classification: this_product vs everything_else
  classifier.fit(train_dir: "data/binary/#{product}")
  classifier.save("models/detect_#{product}")
end

# At inference time, run all detectors
def detect_all_products(image_path)
  detected = []

  products.each do |product|
    detector = Fine::ImageClassifier.load("models/detect_#{product}")
    results = detector.predict(image_path)

    if results.first[:label] == product && results.first[:score] > 0.8
      detected << product
    end
  end

  detected
end
```

## Best Practices

### Image Collection

1. **Variety is key**: Same object, different conditions
2. **Real-world context**: Objects in actual use, not just product shots
3. **Scale variation**: Close-ups and distant shots
4. **Partial visibility**: Objects partially obscured (if that's realistic)

### Data Augmentation

Enable augmentation for more robust recognition:

```ruby
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-384") do |config|
  config.augmentation do |aug|
    aug.random_horizontal_flip = true
    aug.random_rotation = 15          # Degrees
    aug.color_jitter = { brightness: 0.2, contrast: 0.2 }
  end
end
```

### Confidence Thresholds

Set appropriate thresholds based on your use case:

```ruby
results = recognizer.predict(image)
confidence = results.first[:score]

case
when confidence > 0.9
  # High confidence - safe to act automatically
when confidence > 0.7
  # Medium confidence - show to user for confirmation
else
  # Low confidence - likely not a known object
end
```

## Example: Product Catalog Recognition

```ruby
# Train on your product catalog
catalog_recognizer = Fine::ImageClassifier.new("google/siglip2-base-patch16-384") do |config|
  config.epochs = 10
  config.batch_size = 8
  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch + 1}: val_accuracy=#{(metrics[:val_accuracy] * 100).round(1)}%"
  end
end

catalog_recognizer.fit(
  train_dir: "catalog/train",
  val_dir: "catalog/val"
)

catalog_recognizer.save("models/catalog_v1")

# Use in production
def identify_product(photo_path)
  recognizer = Fine::ImageClassifier.load("models/catalog_v1")
  results = recognizer.predict(photo_path, top_k: 3)

  {
    product_id: results.first[:label],
    confidence: results.first[:score],
    alternatives: results[1..2]
  }
end
```
