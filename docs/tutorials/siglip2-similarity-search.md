# Using SigLIP2 for Image Similarity Search

Find visually similar images using learned embeddings.

## When to Use This

- Find similar products in a catalog
- Detect near-duplicates
- Build "more like this" features
- Visual search engines

## How It Works

Instead of classifying images, we extract the embedding vectors from SigLIP2 and compare them. Similar images have similar embeddings.

## Extracting Embeddings

```ruby
require "fine"

# Load a pre-trained model (no fine-tuning needed for general similarity)
# Or load your fine-tuned model for domain-specific similarity
model = Fine::Models::SigLIP2ForImageClassification.from_pretrained(
  "google/siglip2-base-patch16-224",
  num_labels: 1  # Dummy value, we won't use the classifier
)

# Get the encoder only
encoder = model.encoder

# Prepare image transform
transforms = Fine::Transforms::Compose.new([
  Fine::Transforms::Resize.new(224),
  Fine::Transforms::ToTensor.new,
  Fine::Transforms::Normalize.new
])

def get_embedding(encoder, transforms, image_path)
  image = Vips::Image.new_from_file(image_path, access: :sequential)
  tensor = transforms.call(image).unsqueeze(0)  # Add batch dimension

  encoder.eval
  Torch.no_grad do
    encoder.call(tensor).squeeze.to_a
  end
end

# Extract embeddings
embedding1 = get_embedding(encoder, transforms, "image1.jpg")
embedding2 = get_embedding(encoder, transforms, "image2.jpg")
```

## Computing Similarity

```ruby
def cosine_similarity(a, b)
  dot = a.zip(b).sum { |x, y| x * y }
  norm_a = Math.sqrt(a.sum { |x| x * x })
  norm_b = Math.sqrt(b.sum { |x| x * x })
  dot / (norm_a * norm_b)
end

similarity = cosine_similarity(embedding1, embedding2)
puts "Similarity: #{(similarity * 100).round(1)}%"
# > 90% = very similar
# 70-90% = somewhat similar
# < 70% = different
```

## Building a Search Index

For searching through many images:

```ruby
class ImageSearchIndex
  def initialize(encoder, transforms)
    @encoder = encoder
    @transforms = transforms
    @embeddings = {}
  end

  def add(image_path)
    @embeddings[image_path] = get_embedding(image_path)
  end

  def search(query_path, top_k: 5)
    query_emb = get_embedding(query_path)

    results = @embeddings.map do |path, emb|
      { path: path, score: cosine_similarity(query_emb, emb) }
    end

    results.sort_by { |r| -r[:score] }.first(top_k)
  end

  private

  def get_embedding(path)
    image = Vips::Image.new_from_file(path, access: :sequential)
    tensor = @transforms.call(image).unsqueeze(0)

    @encoder.eval
    Torch.no_grad do
      @encoder.call(tensor).squeeze.to_a
    end
  end

  def cosine_similarity(a, b)
    dot = a.zip(b).sum { |x, y| x * y }
    norm_a = Math.sqrt(a.sum { |x| x * x })
    norm_b = Math.sqrt(b.sum { |x| x * x })
    dot / (norm_a * norm_b)
  end
end

# Usage
index = ImageSearchIndex.new(encoder, transforms)

# Index your images
Dir.glob("catalog/*.jpg").each { |path| index.add(path) }

# Search
results = index.search("query_image.jpg", top_k: 10)
results.each do |result|
  puts "#{result[:path]}: #{(result[:score] * 100).round}% similar"
end
```

## Domain-Specific Similarity

For better results on your specific domain, fine-tune first:

```ruby
# Fine-tune on your domain
classifier = Fine::ImageClassifier.new("google/siglip2-base-patch16-224")
classifier.fit(train_dir: "my_domain/train", epochs: 3)
classifier.save("models/my_domain")

# Load the encoder from fine-tuned model
model = Fine::Models::SigLIP2ForImageClassification.load("models/my_domain")
encoder = model.encoder

# Now use this encoder for similarity search
# It will be better at distinguishing items in your domain
```

## Performance Tips

For large catalogs (10k+ images):
- Pre-compute and cache embeddings
- Use approximate nearest neighbor libraries (e.g., Annoy, Faiss via FFI)
- Batch embedding computation for faster indexing
