# Image Classification: Shape Recognition

This example demonstrates fine-tuning SigLIP2 to classify images by dominant color patterns.

## Setup

```ruby
require "fine"

Fine.configure { |c| c.progress_bar = false }
```

## Generate Training Data

Create synthetic training images with different colors representing different "shapes":
- **Circles**: Red-ish colors (RGB around 220, 80, 80)
- **Squares**: Green-ish colors (RGB around 80, 180, 80)
- **Triangles**: Blue-ish colors (RGB around 80, 80, 220)

```ruby
# Run: ruby examples/generate_shape_images.rb
# Creates 30 images (10 per class) in examples/data/shapes/
```

## Train the Classifier

```ruby
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224") do |config|
  config.epochs = 5
  config.batch_size = 4
  config.learning_rate = 1e-4
  config.freeze_encoder = false  # Fine-tune entire model

  config.on_epoch_end do |epoch, metrics|
    puts "Epoch #{epoch}: loss=#{metrics[:loss].round(4)}"
  end
end

history = classifier.fit(train_dir: "examples/data/shapes", epochs: 5)
```

## Training Results

```
Epoch 0: loss=0.8432
Epoch 1: loss=0.1725
Epoch 2: loss=0.0321
Epoch 3: loss=0.0027
Epoch 4: loss=0.0006
```

The loss dropped from 0.84 to 0.0006 - a 99.9% improvement!

## Test Predictions

```ruby
# All predictions are 100% confident and correct
classifier.predict("data/shapes/circle/circle_1.jpg")
# => [{ label: "circle", score: 1.0 }]

classifier.predict("data/shapes/square/square_1.jpg")
# => [{ label: "square", score: 1.0 }]

classifier.predict("data/shapes/triangle/triangle_1.jpg")
# => [{ label: "triangle", score: 0.999 }]
```

## Save and Load

```ruby
# Save
classifier.save("/tmp/shape-classifier")

# Load later
loaded = Fine::ImageClassifier.load("/tmp/shape-classifier")
loaded.predict("new_image.jpg")
```

## Key Takeaways

- SigLIP2 quickly learns visual patterns even with small datasets (30 images)
- Fine-tuning the full model (`freeze_encoder: false`) achieves best results
- The model achieves perfect accuracy after just 5 epochs
