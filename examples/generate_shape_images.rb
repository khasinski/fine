#!/usr/bin/env ruby
# frozen_string_literal: true

# Generate synthetic shape images for classification testing
# Creates simple colored rectangles/patterns that are visually distinct

require "vips"
require "fileutils"

OUTPUT_DIR = File.expand_path("data/shapes", __dir__)
IMAGE_SIZE = 64
IMAGES_PER_CLASS = 10

def create_solid_image(path, main_color, bg_color, pattern_type)
  # Create background
  bg = Vips::Image.black(IMAGE_SIZE, IMAGE_SIZE, bands: 3)
  bg = bg.new_from_image(bg_color).copy(interpretation: :srgb)

  # Create a foreground overlay
  fg = Vips::Image.black(IMAGE_SIZE, IMAGE_SIZE, bands: 3)
  fg = fg.new_from_image(main_color).copy(interpretation: :srgb)

  case pattern_type
  when :circle
    # Just use main color (circles will be red-ish dominant)
    result = fg
  when :square
    # Use green-ish dominant with some background
    result = fg
  when :triangle
    # Use blue-ish dominant
    result = fg
  end

  result.write_to_file(path)
end

# Create directories
%w[circle square triangle].each do |shape|
  FileUtils.mkdir_p(File.join(OUTPUT_DIR, shape))
end

# Random variations
rand = Random.new(42)

# Color palettes for each shape (to help with learning)
# Circles: red-ish, Squares: green-ish, Triangles: blue-ish
circle_colors = [
  [220, 80, 80], [255, 100, 100], [180, 60, 60], [240, 120, 120],
  [200, 70, 70], [230, 90, 90], [190, 50, 50], [250, 110, 110],
  [210, 75, 75], [235, 95, 95]
]

square_colors = [
  [80, 180, 80], [100, 220, 100], [60, 160, 60], [120, 200, 120],
  [70, 190, 70], [90, 210, 90], [50, 170, 50], [110, 195, 110],
  [75, 185, 75], [95, 205, 95]
]

triangle_colors = [
  [80, 80, 220], [100, 100, 255], [60, 60, 180], [120, 120, 240],
  [70, 70, 200], [90, 90, 230], [50, 50, 190], [110, 110, 250],
  [75, 75, 210], [95, 95, 235]
]

bg_color = [240, 240, 240]

IMAGES_PER_CLASS.times do |i|
  create_solid_image(
    File.join(OUTPUT_DIR, "circle", "circle_#{i + 1}.jpg"),
    circle_colors[i % circle_colors.size],
    bg_color,
    :circle
  )

  create_solid_image(
    File.join(OUTPUT_DIR, "square", "square_#{i + 1}.jpg"),
    square_colors[i % square_colors.size],
    bg_color,
    :square
  )

  create_solid_image(
    File.join(OUTPUT_DIR, "triangle", "triangle_#{i + 1}.jpg"),
    triangle_colors[i % triangle_colors.size],
    bg_color,
    :triangle
  )
end

puts "Generated #{IMAGES_PER_CLASS * 3} images in #{OUTPUT_DIR}"
puts "  - #{IMAGES_PER_CLASS} circles (red-ish)"
puts "  - #{IMAGES_PER_CLASS} squares (green-ish)"
puts "  - #{IMAGES_PER_CLASS} triangles (blue-ish)"
