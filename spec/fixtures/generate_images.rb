# frozen_string_literal: true

# Run this script to generate test images:
# ruby spec/fixtures/generate_images.rb

require "vips"

def create_test_image(path, color)
  # Create a 32x32 solid color image
  image = Vips::Image.black(32, 32).add(color)
  image = image.copy(interpretation: :srgb)
  image.write_to_file(path)
  puts "Created: #{path}"
end

# Create cat images (reddish)
2.times do |i|
  create_test_image(
    File.expand_path("images/cat/cat_#{i + 1}.jpg", __dir__),
    [200 + i * 20, 100, 100]
  )
end

# Create dog images (bluish)
2.times do |i|
  create_test_image(
    File.expand_path("images/dog/dog_#{i + 1}.jpg", __dir__),
    [100, 100, 200 + i * 20]
  )
end

puts "Done generating test images!"
